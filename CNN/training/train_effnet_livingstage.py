import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
DATA_DIR = "../inat_data_living"
CSV_PATH = "../inat_data/dataset_metadata.csv"
BATCH_SIZE = 32
EPOCHS = 25
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAXON_EMBED_DIM = 32

# === TAXON GROUP HANDLING ===
df_meta = pd.read_csv(CSV_PATH)
living_df = df_meta[df_meta["stage"].isin(["pre-moult", "moulting", "post-moult"])]
taxon_to_idx = {tg: i for i, tg in enumerate(sorted(living_df["taxon_group"].unique()))}
idx_to_taxon = {i: tg for tg, i in taxon_to_idx.items()}


# === Class-specific transforms for data augmentation ===
class ClassAwareTransform:
    def __init__(self):
        self.transforms = {
            'pre-moult': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'moulting': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
            'post-moult': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.85, 1.2)),
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
        }

    def __call__(self, img, label):
        class_name = train_dataset.classes[label]
        return self.transforms[class_name](img)


# === DATASET CUSTOM ===
class TaxonImageFolder(datasets.ImageFolder):
    def __init__(self, root, metadata_df, taxon_to_idx, transform=None):
        super().__init__(root, transform=lambda x: x)  # disabilitiamo temporaneamente transform
        self.metadata_df = metadata_df
        self.taxon_to_idx = taxon_to_idx
        self.transform_wrapper = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        obs_id = int(os.path.basename(path).split(".")[0])
        row = self.metadata_df[self.metadata_df["observation_id"] == obs_id].iloc[0]
        taxon_group = row["taxon_group"]
        taxon_id = self.taxon_to_idx[taxon_group]
        if self.transform_wrapper:
            img = self.transform_wrapper(img, label)
        return img, label, taxon_id


# === LOAD DATASET ===
train_df = living_df[living_df["split"] == "train"]
val_df = living_df[living_df["split"] == "val"]

train_dataset = TaxonImageFolder(os.path.join(DATA_DIR, "train"), train_df, taxon_to_idx, transform=ClassAwareTransform())
val_dataset = TaxonImageFolder(os.path.join(DATA_DIR, "val"), val_df, taxon_to_idx, transform=ClassAwareTransform())

# === WEIGHT CLASSES ===
targets = [s[1] for s in train_dataset.samples]
class_sample_count = np.bincount(targets)
weights = 1. / class_sample_count
samples_weights = [weights[t] for t in targets]
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# === MODEL WITH EMBED ===
class CustomEffNet(nn.Module):
    def __init__(self, num_classes, taxon_embed_dim=32):
        super().__init__()
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.taxon_embed = nn.Embedding(len(taxon_to_idx), taxon_embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(1280 + taxon_embed_dim),
            nn.Dropout(0.4),
            nn.Linear(1280 + taxon_embed_dim, num_classes)
        )

    def forward(self, x, taxon_id):
        feats = self.backbone(x).squeeze(-1).squeeze(-1)
        taxon_vec = self.taxon_embed(taxon_id)
        x = torch.cat([feats, taxon_vec], dim=1)
        return self.classifier(x)


model = CustomEffNet(num_classes=3, taxon_embed_dim=TAXON_EMBED_DIM).to(DEVICE)

#Freeze starting taxon_embed
for p in model.taxon_embed.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# === TRAINING ===
best_acc = 0
epochs_no_improve = 0
top_epochs = []

os.makedirs("models", exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()

    # Unfreeze taxon_embed after 10 ephocs
    if epoch == 11:
        for p in model.taxon_embed.parameters():
            p.requires_grad = True

    running_loss = 0.0
    for inputs, labels, taxon_ids in tqdm(train_loader, desc=f"Epoch {epoch}"):
        inputs, labels, taxon_ids = inputs.to(DEVICE), labels.to(DEVICE), taxon_ids.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs, taxon_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # === VALIDATION ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels, taxon_ids in val_loader:
            inputs, labels, taxon_ids = inputs.to(DEVICE), labels.to(DEVICE), taxon_ids.to(DEVICE)
            outputs = model(inputs, taxon_ids)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total

    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}")

    top_epochs.append((epoch, acc))
    top_epochs = sorted(top_epochs, key=lambda x: x[1], reverse=True)[:3]

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../models/efficientnet_living_best.pth")
        print("New best model saved")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("⏹Early stopping")
            break

# === REPORT ===
print("\nTop 3 validation epochs:")
for ep, score in top_epochs:
    print(f"Epoch {ep}: {score:.4f}")
