import os
import joblib
import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import cv2

# === CONFIGURATION ===
YOLO_MODEL_PATH = "../training/runs/detect/yolo_moult/weights/best.pt"
XGB_MODEL_PATH = "../models/xgboost_stage.pkl"
ENCODER_PATH = "../models/label_encoder.pkl"
TAXON_GROUPS = ["Crustacea", "Hexapoda", "Chelicerata", "Myriapoda"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load models ===
yolo_model = YOLO(YOLO_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# === Feature extraction from image and bounding boxes ===
def extract_features(image_path, org_box, ex_box, taxon_group):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    feat = {}

    if org_box and ex_box:
        x1_o, y1_o, x2_o, y2_o = org_box
        x1_e, y1_e, x2_e, y2_e = ex_box

        feat["x_organism"] = x1_o
        feat["y_organism"] = y1_o
        feat["x_exuviae"] = x1_e
        feat["y_exuviae"] = y1_e
        feat["h_exuviae"] = y2_e - y1_e

        cx_o = (x1_o + x2_o) / 2
        cy_o = (y1_o + y2_o) / 2
        cx_e = (x1_e + x2_e) / 2
        cy_e = (y1_e + y2_e) / 2
        feat["dist_centroids"] = np.sqrt((cx_o - cx_e) ** 2 + (cy_o - cy_e) ** 2)

        inter_x1 = max(x1_o, x1_e)
        inter_y1 = max(y1_o, y1_e)
        inter_x2 = min(x2_o, x2_e)
        inter_y2 = min(y2_o, y2_e)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_o = (x2_o - x1_o) * (y2_o - y1_o)
        area_e = (x2_e - x1_e) * (y2_e - y1_e)
        union_area = area_o + area_e - inter_area
        feat["box_overlap"] = inter_area / union_area if union_area > 0 else 0
    else:
        for key in ["x_organism", "y_organism", "x_exuviae", "y_exuviae", "h_exuviae", "dist_centroids", "box_overlap"]:
            feat[key] = -1

    # === Binary feature indicating presence of only exuviae ===
    feat["only_exuviae"] = 1 if ex_box and not org_box else 0

    # === Compute color features from organism box ===
    if org_box:
        x1, y1, x2, y2 = map(int, org_box)
        patch = img_np[y1:y2, x1:x2]
        if patch.size > 0:
            feat["org_mean_g"] = np.mean(patch[..., 1])
            feat["org_mean_gray"] = np.mean(np.mean(patch, axis=2))
        else:
            feat["org_mean_g"] = -1
            feat["org_mean_gray"] = -1
    else:
        feat["org_mean_g"] = -1
        feat["org_mean_gray"] = -1

    # === One-hot encode taxon group ===
    for tg in TAXON_GROUPS:
        feat[f"taxon_group_{tg}"] = 1 if tg == taxon_group else 0

    df = pd.DataFrame([feat])
    df = df[xgb_model.get_booster().feature_names]  # Ensure correct feature order
    return df

# === Full prediction pipeline ===
def predict_stage_from_image(image_path: str, taxon_group: str):
    results = yolo_model.predict(image_path, device=DEVICE, conf=0.25, verbose=False)
    boxes = {"organism": None, "exuviae": None}
    confs = {"organism": 0, "exuviae": 0}

    for r in results:
        for b in r.boxes:
            cls = int(b.cls)
            score = float(b.conf)
            coords = list(map(int, b.xyxy[0]))
            if cls == 0 and score > confs["organism"]:
                boxes["organism"] = coords
                confs["organism"] = score
            elif cls == 1 and score > confs["exuviae"]:
                boxes["exuviae"] = coords
                confs["exuviae"] = score

    features = extract_features(image_path, boxes["organism"], boxes["exuviae"], taxon_group)
    pred_idx = xgb_model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    prob = max(xgb_model.predict_proba(features)[0])
    return pred_label, prob, boxes

# === Visualization with bounding boxes and prediction ===
def test_and_draw(image_path: str, taxon_group: str, output_path="../results/predictions/test_img_out.jpg"):
    stage, conf, boxes = predict_stage_from_image(image_path, taxon_group)
    image = cv2.imread(image_path)

    if boxes["organism"]:
        x1, y1, x2, y2 = boxes["organism"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, "organism", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if boxes["exuviae"]:
        x1, y1, x2, y2 = boxes["exuviae"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, "exuviae", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    label = f"{stage} ({conf:.2f})"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")
    print(f"Prediction: {stage} (Confidence: {conf:.2f})")

# === Run standalone test ===
if __name__ == "__main__":
    test_and_draw("../../test/cicada.jpg", "Hexapoda")
