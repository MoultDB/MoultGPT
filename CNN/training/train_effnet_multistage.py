import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import random

# === CONFIG ===
DATA_DIR = "../inat_data"
METADATA_CSV = "../inat_data/dataset_metadata.csv"
BATCH_SIZE = 32
EPOCHS = 25
FREEZE_TAXON_EPOCHS = 10
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAXON_EMBED_DIM = 32

# === Helper per augment class-aware ===
def get_class_aware_transform(label_name):
    if label_name == "exuviae":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.GaussianBlur(kernel_size=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    elif label_name == "post-moult":
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, hue=0.05),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])
    elif label_name == "moulting":
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    elif label_name == "pre-moult":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

# === Dataset custom ===
class ClassAwareDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, metadata_csv, split):
        self.base_path = os.path.join(root_dir, split)
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split]
        self.classes_step1 = {"living": 0, "exuviae": 1}
        self.classes_step2 = {"pre-moult": 0, "moulting": 1, "post-moult": 2}
        self.taxon_groups = sorted(self.df["taxon_group"].dropna().unique().tolist())
        self.taxon2idx = {t: i for i, t in enumerate(self.taxon_groups)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_path, row["stage"], row["filename"])
        label_name = row["stage"]
        image = Image.open(img_path).convert("RGB")
        image = get_class_aware_transform(label_name)(image)

        # Step 1 label: exuviae vs living
        label1 = self.classes_step1["exuviae"] if label_name == "exuviae" else self.classes_step1["living"]

        # Step 2 label (solo se living)
        label2 = self.classes_step2[label_name] if label_name != "exuviae" else -1

        # Taxon embedding
        taxon_tensor = torch.tensor(self.taxon2idx.get(row["taxon_group"], 0), dtype=torch.long)

        return image, label1, label2, taxon_tensor

# === Model ===
class MultiTaskEffNet(nn.Module):
    def __init__(self, num_taxa, embed_dim=32):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = base.features
        self.pooling = base.avgpool
        self.dropout = nn.Dropout(0.4)
        self.taxon_embed = nn.Sequential(
            nn.Embedding(num_taxa, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.2)
        )
        self.classifier1 = nn.Linear(1280 + embed_dim, 2)  # step1: exuviae/living
        self.classifier2 = nn.Linear(1280 + embed_dim, 3)  # step2: pre/moulting/post

    def forward(self, x, taxon_id):
        feat = self.feature_extractor(x)
        feat = self.pooling(feat).squeeze(-1).squeeze(-1)
        taxon = self.taxon_embed(taxon_id)
        xcat = torch.cat((feat, taxon), dim=1)
        out1 = self.classifier1(self.dropout(xcat))
        out2 = self.classifier2(self.dropout(xcat))
        return out1, out2

# === Load datasets
train_dataset = ClassAwareDataset(DATA_DIR, METADATA_CSV, "train")
val_dataset = ClassAwareDataset(DATA_DIR, METADATA_CSV, "val")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = MultiTaskEffNet(num_taxa=len(train_dataset.taxon2idx), embed_dim=TAXON_EMBED_DIM).to(DEVICE)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0
epochs_no_improve = 0
top_epochs = []

os.makedirs("models", exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    if epoch == FREEZE_TAXON_EPOCHS + 1:
        for p in model.taxon_embed.parameters():
            p.requires_grad = True

    running_loss = 0
    for images, label1, label2, taxon in train_loader:
        images = images.to(DEVICE)
        label1 = label1.to(DEVICE)
        label2 = label2.to(DEVICE)
        taxon = taxon.to(DEVICE)

        optimizer.zero_grad()
        out1, out2 = model(images, taxon)
        loss = criterion1(out1, label1) + criterion2(out2, label2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # === Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, label1, label2, taxon in val_loader:
            images = images.to(DEVICE)
            label1 = label1.to(DEVICE)
            taxon = taxon.to(DEVICE)
            out1, out2 = model(images, taxon)
            preds = out1.argmax(dim=1)
            correct += (preds == label1).sum().item()
            total += label1.size(0)

    acc = correct / total
    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Step1 Val Acc: {acc:.4f}")

    top_epochs.append((epoch, acc))
    top_epochs = sorted(top_epochs, key=lambda x: x[1], reverse=True)[:3]

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../models/effnet_multistage_best.pth")
        print("New best model saved")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("⏹Early stopping")
            break

# === Report top epochs
print("\\nTop 3 validation epochs:")
for ep, score in top_epochs:
    print(f"Epoch {ep}: {score:.4f}")

