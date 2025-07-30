import os
import shutil
import pandas as pd
from PIL import Image

CSV_PATH = "../../data/annotated_dataset.csv"
IMAGES_ROOT = "../../data/inat"
YOLO_ROOT = "../../data/yolo"

df = pd.read_csv(CSV_PATH)
class_map = {"organism": 0, "exuviae": 1}

def convert_bbox_to_yolo(size, box):
    img_w, img_h = size
    x, y, w, h = box
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h

for subset in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_ROOT, "images", subset), exist_ok=True)
    os.makedirs(os.path.join(YOLO_ROOT, "labels", subset), exist_ok=True)

skipped = []
for _, row in df.iterrows():
    filename = row["filename"]
    subset = row["split"]
    image_path = os.path.join(IMAGES_ROOT, subset, row["stage"], filename)
    out_img = os.path.join(YOLO_ROOT, "images", subset, filename)
    out_lbl = os.path.join(YOLO_ROOT, "labels", subset, filename.replace(".jpg", ".txt"))

    if not os.path.exists(image_path):
        skipped.append(filename)
        continue

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except:
        skipped.append(filename)
        continue

    lines = []
    if not pd.isna(row["x_organism"]):
        xc, yc, wn, hn = convert_bbox_to_yolo((img_w, img_h), (row["x_organism"], row["y_organism"], row["w_organism"], row["h_organism"]))
        lines.append(f"{class_map['organism']} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    if not pd.isna(row["x_exuviae"]):
        xc, yc, wn, hn = convert_bbox_to_yolo((img_w, img_h), (row["x_exuviae"], row["y_exuviae"], row["w_exuviae"], row["h_exuviae"]))
        lines.append(f"{class_map['exuviae']} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    if not lines:
        skipped.append(filename)
        continue

    with open(out_lbl, "w") as f:
        f.write("\n".join(lines))

    shutil.copy(image_path, out_img)

print(f"Conversione completata. Skipped {len(skipped)} immagini.")
