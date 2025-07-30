import os
import urllib.request
import subprocess

# === Configuration ===
MODEL_NAME = "yolov8m.pt"
DOWNLOAD_URL = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{MODEL_NAME}"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

YOLO_DATA_PATH = os.path.join(BASE_DIR, "data", "yolo", "moulting.yaml")
EXPERIMENT_NAME = "yolo_moult"
IMG_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16

def download_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"[✓] Model already exists: {MODEL_PATH}")
        return
    print(f"Downloading {MODEL_NAME} to {MODEL_PATH}...")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, MODEL_PATH)
        print(f"[✓] Download completed.")
    except Exception as e:
        print(f"[✗] Failed to download model: {e}")
        exit(1)

def train_model():
    print(f"Starting YOLOv8 training...")
    cmd = [
        "yolo",
        "task=detect",
        "mode=train",
        f"model={MODEL_PATH}",
        f"data={YOLO_DATA_PATH}",
        f"epochs={EPOCHS}",
        f"imgsz={IMG_SIZE}",
        f"batch={BATCH_SIZE}",
        f"name={EXPERIMENT_NAME}"
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("[✓] Training completed.")
    else:
        print("[✗] Training failed.")

if __name__ == "__main__":
    download_model()
    train_model()
