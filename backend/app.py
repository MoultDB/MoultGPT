import os
import sys
import torch
import json
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# === Local imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from CNN.scripts.predict.image_predictor import predict_stage_from_image as predict_image
from paper_handler.processor import input_to_text
from backend.preprocessing import extract_relevant_sentences

# === Init Flask app ===
print("[BOOT] Initializing Flask backend...")
app = Flask(__name__)
CORS(app)

# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = "/reference/LLMs/Mistral_AI/mistral-7B-Instruct-v0.3-hf/"
LORA_PATH = os.path.join(BASE_DIR, "output", "lora_mistral")

# === Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

USE_LORA = False
model = PeftModel.from_pretrained(base_model, LORA_PATH) if USE_LORA else base_model
model.eval()

print(f"[BOOT] Loaded base model: {MODEL_PATH}")
print(f"[BOOT] LoRA enabled: {USE_LORA}")

# === Routes ===

@app.route("/", methods=["GET"])
def root():
    return "LLM backend is running"

@app.route("/query", methods=["POST"])
def handle_query():
    print("[QUERY] Incoming request to /query")
    try:
        data = request.form.to_dict()
        doi = data.get("doi", "").strip()
        prompt = data.get("prompt", "").strip()
        raw_text = data.get("text", "").strip()
        uploaded_file = request.files.get("file")

        if not prompt:
            return jsonify({"response": "Missing query prompt."}), 400

        # Input: DOI, file, or raw text
        if doi:
            full_text = input_to_text(doi=doi, email="your@email.com")
        elif uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                uploaded_file.save(tmp.name)
                full_text = input_to_text(pdf_path=tmp.name)
                os.unlink(tmp.name)
        elif raw_text:
            full_text = raw_text
        else:
            return jsonify({"response": "No valid input provided."}), 400

        if not full_text or len(full_text.strip()) < 100:
            return jsonify({"response": "Could not extract meaningful content."}), 500

        summary = extract_relevant_sentences(full_text)
        print(f"[INFO] Extracted {summary.count(chr(10))} relevant lines")

        fixed_prompt = (
            "You are a scientific assistant specialized in arthropod moulting.\n"
            "Extract only the specific biological trait requested in the prompt.\n"
            "Return only clean, concise YAML. Do not include any explanation, metadata, or extra text."
        )
        combined_prompt = f"<s>[INST] {fixed_prompt}\n\n{summary.strip()}\n\n{prompt.strip()} [/INST]"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(combined_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        print("[INFO] Running LLM generation...")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output.split("[/INST]")[-1].strip()
        return jsonify({"response": response})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"response": f"Internal error: {str(e)}"}), 500

@app.route("/feedback", methods=["POST"])
def handle_feedback():
    print("[FEEDBACK] Received user feedback")
    try:
        data = request.json
        feedback_entry = {
            "prompt": data.get("prompt", "").strip(),
            "model_response": data.get("response", "").strip(),
            "user_feedback": data.get("feedback", "").strip(),
            "source": data.get("source", "manual_feedback")
        }

        if not feedback_entry["prompt"] or not feedback_entry["model_response"]:
            return jsonify({"status": "Missing required fields"}), 400

        feedback_path = os.path.join(BASE_DIR, "output", "user_feedback.jsonl")
        with open(feedback_path, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")

        return jsonify({"status": "Feedback saved"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": f"Internal error: {str(e)}"}), 500

@app.route("/predict_image", methods=["POST"])
def handle_image_prediction():
    print("[IMAGE] Incoming image prediction request")
    try:
        if "image" not in request.files or "taxon_id" not in request.form:
            return jsonify({"error": "Missing image or taxon_id"}), 400

        image_file = request.files["image"]
        taxon_id = int(request.form["taxon_id"])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_file.save(tmp.name)
            pred_label, prob, boxes = predict_image(tmp.name, taxon_id)
            os.unlink(tmp.name)

        # Clean result
        result_clean = {
            "label": pred_label,
            "confidence": float(prob),
            "boxes": {k: v if v is not None else [] for k, v in boxes.items()}
        }

        return jsonify(result_clean)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

# === Start ===
if __name__ == "__main__":
    print("[BOOT] Running backend on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001)
