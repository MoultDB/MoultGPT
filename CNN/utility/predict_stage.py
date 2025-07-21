import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import torch.nn.functional as F

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
        self.classifier1 = nn.Linear(1280 + embed_dim, 2)  # step1
        self.classifier2 = nn.Linear(1280 + embed_dim, 3)  # step2

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

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--taxon", type=int, required=True)
args = parser.parse_args()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

try:
    image = Image.open(args.image).convert("RGB")
except Exception as e:
    print(f"Error in loading pictures: {e}")
    exit(1)

img_tensor = transform(image).unsqueeze(0).to(DEVICE)
taxon_tensor = torch.tensor([args.taxon], dtype=torch.long).to(DEVICE)

# === LOAD MODELS ===
try:
    step1_model = MultiTaskEffNet(num_taxa=4, embed_dim=32).to(DEVICE)
    step1_model.load_state_dict(torch.load("models/effnet_multistage_best.pth", map_location=DEVICE))
    step1_model.eval()

    step2_model = CustomEffNet(num_classes=3, num_taxa=4, embed_dim=32).to(DEVICE)
    step2_model.load_state_dict(torch.load("models/efficientnet_living_best.pth", map_location=DEVICE))
    step2_model.eval()

except Exception as e:
    print(f"Error in loading models: {e}")
    exit(1)

# === STEP 1: exuviae vs living ===
with torch.no_grad():
    out1, _ = step1_model(img_tensor, taxon_tensor)
    probs1 = F.softmax(out1, dim=1)
    pred1 = probs1.argmax(dim=1).item()
    conf1 = probs1[0, pred1].item()

if pred1 == 1:
    print(f"Prediction: exuviae (confidence: {conf1*100:.1f}%)")
else:
    # STEP 2
    with torch.no_grad():
        out2 = step2_model(img_tensor, taxon_tensor)
        probs2 = F.softmax(out2, dim=1)
        pred2 = probs2.argmax(dim=1).item()
        conf2 = probs2[0, pred2].item()
        label_map = {0: "pre-moult", 1: "moulting", 2: "post-moult"}
        print(f"Prediction: {label_map[pred2]} (confidence: {conf2*100:.1f}%)")

print(f"Taxon: {TAXON_LABELS.get(args.taxon, 'Unknown')} (id: {args.taxon})")

