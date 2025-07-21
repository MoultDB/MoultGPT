import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# === CONFIG ===
CSV_PATH = "../inat_data/dataset_metadata.csv"
SRC_IMG_DIR = Path("../inat_data")  
OUT_DIR = Path("../inat_data_living")  
LIVING_CLASSES = ["pre-moult", "moulting", "post-moult"]
VAL_RATIO = 0.2

# === Rwad Metadata ===
df = pd.read_csv(CSV_PATH)
df = df[df["stage"].isin(LIVING_CLASSES)]

# === Split train/val ===
train_df, val_df = train_test_split(df, test_size=VAL_RATIO, stratify=df["stage"], random_state=42)

# === FUNCTION FOR IMAGE-COPY ===
def copy_images(subset_df, split):
    for _, row in subset_df.iterrows():
        cls = row["stage"]
        obs_id = row["observation_id"]
        filename = f"{obs_id}.jpg"
        src_path = SRC_IMG_DIR / split / cls / filename
        dest_dir = OUT_DIR / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        if src_path.exists():
            shutil.copy2(src_path, dest_path)

# === Esegui copia
copy_images(train_df, "train")
copy_images(val_df, "val")
