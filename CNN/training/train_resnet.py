import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import copy

# === Config ===
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = "../inat_data"
NUM_CLASSES = 3  # Excluding pre-moult
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Mixup ===
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# === Augmentations ===
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.4, 0.4, 0.3, 0.05),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomAffine(15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Dataset filter ===
class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, exclude_class="pre-moult"):
        super().__init__(root, transform=transform)
        idx = self.class_to_idx.get(exclude_class)
        if idx is not None:
            self.samples = [s for s in self.samples if s[1] != idx]
            self.targets = [s[1] for s in self.samples]

# === Load datasets ===
train_dataset = FilteredImageFolder(os.path.join(DATA_DIR, "train"), train_transform)
val_dataset = FilteredImageFolder(os.path.join(DATA_DIR, "val"), val_transform)

# === Weighted sampling ===
class_counts = np.bincount(train_dataset.targets)
weights = 1. / class_counts
sample_weights = [weights[t] for t in train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# === Train loop with early stopping ===
best_model_wts = None
best_acc = 0.0
top_epochs = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # === Validation ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    top_epochs.append((epoch + 1, val_acc))

# === Save best model ===
os.makedirs("models", exist_ok=True)
torch.save(best_model_wts, "../models/resnet18_moulting_best.pth")
print("\n Best model saved to models/resnet18_moulting_best.pth")

# === Report Top 3 ===
top_epochs.sort(key=lambda x: x[1], reverse=True)
print(" Top 3 epochs:")
for ep, acc in top_epochs[:3]:
    print(f"Epoch {ep}: {acc:.4f}")
