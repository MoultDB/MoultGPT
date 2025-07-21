import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
import pandas as pd
import shutil
from collections import defaultdict
from tqdm import tqdm

# === Config ===
VAL_DIR = "../inat_data/val"
CSV_METADATA = "../inat_data/dataset_metadata.csv"
MODEL_PATH = "../models/efficientnet_b0_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "../errors_analysis"

# === Load metadata with taxon_group info
df = pd.read_csv(CSV_METADATA)
filename_to_taxon = dict(zip(df["filename"], df["taxon_group"]))

# === Label encoder for taxon group
TAXON_GROUPS = sorted(set(filename_to_taxon.values()))
taxon_to_id = {name: i for i, name in enumerate(TAXON_GROUPS)}

# === Load model (same structure as in train)
class CustomEffNet(nn.Module):
    def __init__(self, num_classes, taxon_embed_dim=16):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(0.4)
        self.taxon_embed = nn.Embedding(len(taxon_to_id), taxon_embed_dim)
        self.classifier = nn.Linear(1280 + taxon_embed_dim, num_classes)

    def forward(self, x, taxon_group_id):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        t = self.taxon_embed(taxon_group_id)
        x = torch.cat((x, t), dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# === Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# === Load model
model = CustomEffNet(num_classes=len(val_dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Create output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Predict and copy misclassified images
class_idx_to_name = {v: k for k, v in val_dataset.class_to_idx.items()}
errors = defaultdict(int)

with torch.no_grad():
    for img, label in tqdm(val_loader, desc="Analyzing errors"):
        path, _ = val_dataset.samples[errors[None]]
        filename = os.path.basename(path)
        taxon_name = filename_to_taxon.get(filename, None)

        if taxon_name is None:
            continue

        taxon_id = torch.tensor([taxon_to_id[taxon_name]], device=DEVICE)
        img = img.to(DEVICE)
        outputs = model(img, taxon_id)
        pred = outputs.argmax(dim=1).item()

        true_label = label.item()
        if pred != true_label:
            src_path = path
            dst_dir = os.path.join(OUTPUT_DIR, f"true_{class_idx_to_name[true_label]}_pred_{class_idx_to_name[pred]}")
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dst_dir, filename))
            errors[(true_label, pred)] += 1

# === Report
print("\\nDone. Misclassifications copied to:", OUTPUT_DIR)
print("Counts per (true, pred) class:")
for (t, p), count in sorted(errors.items()):
    print(f"True: {class_idx_to_name[t]:<10} | Pred: {class_idx_to_name[p]:<10} | Count: {count}")

