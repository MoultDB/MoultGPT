# CNN/image_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# === CONFIG ===
TAXON_LABELS = {0: "Arachnida", 1: "Crustacea", 2: "Hexapoda", 3: "Myriapoda"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODELS ===
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
        self.classifier1 = nn.Linear(1280 + embed_dim, 2)
        self.classifier2 = nn.Linear(1280 + embed_dim, 3)

    def forward(self, x, taxon_id):
        feat = self.feature_extractor(x)
        feat = self.pooling(feat).squeeze(-1).squeeze(-1)
        taxon = self.taxon_embed(taxon_id)
        xcat = torch.cat((feat, taxon), dim=1)
        out1 = self.classifier1(self.dropout(xcat))
        out2 = self.classifier2(self.dropout(xcat))
        return out1, out2

class CustomEffNet(nn.Module):
    def __init__(self, num_classes, num_taxa=4, embed_dim=32):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.taxon_embed = nn.Embedding(num_taxa, embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(1280 + embed_dim),
            nn.Dropout(0.4),
            nn.Linear(1280 + embed_dim, num_classes)
        )

    def forward(self, x, taxon_id):
        feats = self.backbone(x).squeeze(-1).squeeze(-1)
        taxon_vec = self.taxon_embed(taxon_id)
        x = torch.cat([feats, taxon_vec], dim=1)
        return self.classifier(x)

# === LOAD MODELS ONCE ===
step1_model = MultiTaskEffNet(num_taxa=4, embed_dim=32).to(DEVICE)
step1_model.load_state_dict(torch.load("CNN/models/effnet_multistage_best.pth", map_location=DEVICE))
step1_model.eval()

step2_model = CustomEffNet(num_classes=3, num_taxa=4, embed_dim=32).to(DEVICE)
step2_model.load_state_dict(torch.load("CNN/models/efficientnet_living_best.pth", map_location=DEVICE))
step2_model.eval()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === PREDICT FUNCTION ===
def predict_image(image_path: str, taxon_id: int) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        taxon_tensor = torch.tensor([taxon_id], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            out1, _ = step1_model(img_tensor, taxon_tensor)
            probs1 = F.softmax(out1, dim=1)
            pred1 = probs1.argmax(dim=1).item()
            conf1 = probs1[0, pred1].item()

        if pred1 == 1:
            return {
                "prediction": "exuviae",
                "confidence": round(conf1 * 100, 1),
                "taxon": TAXON_LABELS.get(taxon_id, "Unknown"),
                "taxon_id": taxon_id
            }

        with torch.no_grad():
            out2 = step2_model(img_tensor, taxon_tensor)
            probs2 = F.softmax(out2, dim=1)
            pred2 = probs2.argmax(dim=1).item()
            conf2 = probs2[0, pred2].item()
            label_map = {0: "pre-moult", 1: "moulting", 2: "post-moult"}

        return {
            "prediction": label_map[pred2],
            "confidence": round(conf2 * 100, 1),
            "taxon": TAXON_LABELS.get(taxon_id, "Unknown"),
            "taxon_id": taxon_id
        }

    except Exception as e:
        return {"error": str(e)}
