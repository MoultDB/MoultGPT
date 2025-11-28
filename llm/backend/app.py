# llm/backend/app.py
"""
MoultGPT — LLM backend

- Accepts DOI, PDF upload or raw text + user prompt.
- Uses the LLM pipeline modules to:
  1) Convert DOI/PDF to text (Unpaywall + GROBID).
  2) Analyze the paper for arthropod moulting relevance:
     - detect arthropod taxa,
     - extract biologically relevant sentences about moulting,
     - decide if the paper is relevant enough.
  3) Route or block user queries according to scope:
     - arthropod moulting ONLY; vertebrate moulting and off-topic questions rejected.
  4) Query a local Mistral 7B (optionally with LoRA) and return YAML only.

Image prediction is NOT handled here (that's in the vision backend).

Run locally (for cluster):
    cd llm/backend
    python app.py

Environment variables:
    PORT             (default: 5002)
    MODEL_PATH       (HF/local path to base model)
    LORA_PATH        (path to LoRA weights; optional)
    USE_LORA         ("true"/"false", default: false)
    MAX_INPUT_TOKENS (default: 2048)
    MAX_NEW_TOKENS   (default: 512)
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Optional: PEFT for LoRA
try:
    from peft import PeftModel
except Exception:
    PeftModel = None
    print("[LLM] WARNING: 'peft' not installed. LoRA will be disabled.")

# ────────────────────── Path setup ──────────────────────

# This file lives in llm/backend/app.py
# We add the llm/ root so we can import from llm/pipeline/*
LLM_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from pipeline.processor import input_to_text  # type: ignore  # noqa: E402
from pipeline.gating import (  # type: ignore  # noqa: E402
    analyze_paper_for_moulting,
    route_query_for_paper,
)

# ────────────────────── Config ──────────────────────

PORT = int(os.getenv("PORT", 5002))

# Base model + LoRA
DEFAULT_MODEL_PATH = "/reference/LLMs/Mistral_AI/mistral-7B-Instruct-v0.3-hf/"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
LORA_PATH = os.getenv("LORA_PATH", str(LLM_ROOT / "backend" / "models" / "llm"))
USE_LORA = os.getenv("USE_LORA", "false").lower() == "true"

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 2048))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))

# Device selection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def _load_model_and_tokenizer() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load base Mistral model (4-bit if GPU available) and optional LoRA.
    Designed to run primarily on a GPU node; on pure CPU it will be slow.
    """
    print(f"[LLM] Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    quant_config: Optional[BitsAndBytesConfig] = None
    if DEVICE == "cuda":
        print("[LLM] Using 4-bit quantization (bitsandbytes)")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"[LLM] Loading base model from {MODEL_PATH} on device={DEVICE}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    if USE_LORA:
        if PeftModel is None:
            print(
                "[LLM] WARNING: USE_LORA=true but 'peft' is not available. "
                "Proceeding without LoRA."
            )
            model = base_model
        else:
            if not os.path.isdir(LORA_PATH):
                raise RuntimeError(
                    f"USE_LORA=true but LORA_PATH does not exist: {LORA_PATH}"
                )
            print(f"[LLM] Loading LoRA weights from {LORA_PATH}")
            model = PeftModel.from_pretrained(base_model, LORA_PATH)
    else:
        model = base_model

    # For CPU / MPS, move the model explicitly if not using quantization.
    if DEVICE in ("cpu", "mps") and quant_config is None:
        model.to(DEVICE)

    model.eval()
    return tokenizer, model


# Load model at startup
try:
    TOKENIZER, MODEL = _load_model_and_tokenizer()
    print(f"[LLM] Model loaded successfully on device={DEVICE}")
except Exception as e:
    print(f"[LLM] ERROR loading model: {e}")
    TOKENIZER, MODEL = None, None  # type: ignore[assignment]

# ────────────────────── Flask app ──────────────────────

print("[BOOT] Initializing MoultGPT LLM backend...")
app = Flask(__name__)
CORS(app)

# ────────────────────── Warm-up: taxonomy graph ──────────────────────

print("[BOOT] Warming up taxonomy graph...")

from domain.taxonomy_graph import (  # type: ignore  # noqa: E402,E401
    load_taxonomy,
    build_taxonomy_graph,
    build_name_index,
)

t0_tax = time.time()
load_taxonomy()
build_taxonomy_graph()
build_name_index()
print(f"[BOOT] Taxonomy ready. Warm-up time: {time.time() - t0_tax:.2f}s")


# ────────────────────── Helpers ──────────────────────


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "module": "llm", "device": DEVICE})


def build_fixed_prompt() -> str:
    """
    System prompt specialized for arthropod moulting.
    Later this can be moved to llm/pipeline/prompts.py.
    """
    return (
        "You are a scientific assistant specialized in arthropod moulting.\n"
        "You receive:\n"
        "  1) A set of sentences extracted from a scientific paper, already\n"
        "     filtered to focus on moulting-related content.\n"
        "  2) A user query describing which biological trait to extract.\n\n"
        "Your task:\n"
        "- Extract ONLY the trait requested in the user query.\n"
        "- Ignore all other information.\n"
        "- If the trait is not mentioned or cannot be inferred, say so clearly.\n"
        "- Return the answer as CLEAN YAML only, with no extra prose,\n"
        "  no explanations, and no surrounding text.\n"
    )


def build_inst_prompt(summary: str, user_prompt: str) -> str:
    system_prompt = build_fixed_prompt()
    return (
        f"<s>[INST] {system_prompt.strip()}\n\n"
        f"Context:\n{summary.strip()}\n\n"
        f"User query:\n{user_prompt.strip()}\n[/INST]"
    )


def _ensure_model_loaded():
    if TOKENIZER is None or MODEL is None:
        raise RuntimeError("LLM model is not loaded; check server logs.")


def _extract_full_text_from_request():
    """
    Resolve DOI / file / raw text into a single full_text string.

    Returns:
        full_text: str or None
        source: "doi" | "file" | "text" | None
        saved_pdf_path: str or None
        used_doi: bool
        used_pdf: bool
        used_raw_text: bool
    """
    doi = request.form.get("doi", "").strip()
    raw_text = request.form.get("text", "").strip()
    file = request.files.get("file")

    full_text: Optional[str] = None
    source: Optional[str] = None
    saved_pdf_path: Optional[str] = None

    if doi:
        print(f"[LLM] Using DOI={doi}")
        full_text = input_to_text(doi=doi)
        source = "doi"

    elif file:
        tmp_dir = LLM_ROOT / "data" / "papers_pdf"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / file.filename
        file.save(tmp_path)
        print(f"[LLM] Saved uploaded PDF to {tmp_path}")
        full_text = input_to_text(pdf_path=str(tmp_path))
        source = "file"
        saved_pdf_path = str(tmp_path)

    elif raw_text:
        print("[LLM] Using raw text from request.")
        full_text = raw_text
        source = "text"

    return full_text, source, saved_pdf_path, bool(doi), bool(file), bool(raw_text)


# ────────────────────── Preprocess endpoint ──────────────────────


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """
    Debug / helper endpoint: run only the pre-processing pipeline
    (paper analysis and gating-related stats) without calling the LLM.

    Form fields:
        - doi       (optional)
        - text      (optional, raw text)
        - file      (optional, PDF upload)

    Returns:
        - source: which input was used
        - full_text_chars: length of extracted text
        - full_text_preview: first 1000 chars
        - summary: relevant sentences (what we would feed to the LLM)
        - paper_taxa: arthropod taxa detected in the paper
        - n_summary: number of relevant sentences
        - paper_has_arthropods: bool
        - paper_is_relevant: bool (summary length >= threshold)
    """
    try:
        full_text, source, _, used_doi, used_pdf, used_raw_text = _extract_full_text_from_request()

        if not full_text or len(full_text.strip()) < 100:
            return (
                jsonify(
                    {
                        "error": "Could not extract meaningful content.",
                        "details": "Text too short or empty after extraction.",
                    }
                ),
                500,
            )

        analysis = analyze_paper_for_moulting(full_text)

        return jsonify(
            {
                "source": source,
                "full_text_chars": len(full_text),
                "full_text_preview": full_text[:1000],
                "summary": analysis["summary"],
                "paper_taxa": analysis["paper_taxa"],
                "n_summary": analysis["n_summary"],
                "paper_has_arthropods": analysis["paper_has_arthropods"],
                "paper_is_relevant": analysis["paper_is_relevant"],
                "used_doi": used_doi,
                "used_pdf": used_pdf,
                "used_raw_text": used_raw_text and not used_doi and not used_pdf,
            }
        )

    except Exception as e:
        print(f"[LLM] ERROR in /preprocess: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# ────────────────────── Main LLM endpoint ──────────────────────


@app.route("/query", methods=["POST"])
def query():
    """
    Main LLM endpoint.

    Form fields:
        - doi       (optional, string)
        - text      (optional, raw text)
        - prompt    (required, user query)
        - file      (optional, PDF upload)

    Priority for text source:
        1) doi
        2) file (PDF)
        3) raw text

    Routing logic (in pipeline.gating):
        - Question-level gate:
            * only arthropod moulting questions allowed
            * vertebrate moulting or generic non-moulting questions rejected
        - Paper-level gate:
            * requires enough moulting-related sentences in the paper
    """
    t0 = time.time()

    user_prompt = request.form.get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' in form data."}), 400

    try:
        full_text, source, saved_pdf_path, used_doi, used_pdf, used_raw_text = _extract_full_text_from_request()

        if not full_text or len(full_text.strip()) < 100:
            return (
                jsonify(
                    {
                        "error": "Could not extract meaningful content from the input.",
                        "details": "Text too short or empty after extraction.",
                    }
                ),
                500,
            )

        # Unified paper analysis (taxa + summary + counts)
        analysis = analyze_paper_for_moulting(full_text)
        paper_taxa = analysis["paper_taxa"]
        summary = analysis["summary"]
        n_summary = analysis["n_summary"]

        # Combined routing (question + paper relevance)
        decision = route_query_for_paper(user_prompt, paper_taxa, n_summary)
        print(
            f"[ROUTING] allow={decision['allow']} "
            f"stage={decision['stage']} reason={decision['reason']} "
            f"paper_taxa_count={len(paper_taxa)} n_summary={n_summary}"
        )

        if not decision["allow"]:
            return (
                jsonify(
                    {
                        "error": "out_of_scope",
                        "message": decision["message"],
                        "reason": decision["reason"],
                        "stage": decision["stage"],
                        "paper_taxa": paper_taxa,
                        "n_relevant_sentences": n_summary,
                    }
                ),
                400,
            )

        if not summary or len(summary.strip()) == 0:
            return (
                jsonify(
                    {
                        "error": "No relevant sentences could be extracted.",
                        "details": "The article may not contain moulting-related content.",
                    }
                ),
                500,
            )

        # Make sure model is ready
        _ensure_model_loaded()

        # Build prompt
        combined_prompt = build_inst_prompt(summary, user_prompt)

        # Tokenize
        inputs = TOKENIZER(
            combined_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        if DEVICE != "cpu":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = MODEL.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1,
                eos_token_id=TOKENIZER.eos_token_id,
            )

        output = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)

        # For Mistral-style chat, everything after [/INST] is the assistant reply.
        if "[/INST]" in output:
            response = output.split("[/INST]", 1)[-1].strip()
        else:
            response = output.strip()

        dt = time.time() - t0
        print(f"[LLM] /query completed in {dt:.2f}s (source={source}, pdf={saved_pdf_path})")

        return jsonify(
            {
                "response": response,
                "latency_sec": dt,
                "used_doi": used_doi,
                "used_pdf": used_pdf,
                "used_raw_text": used_raw_text and not used_doi and not used_pdf,
                "routing_stage": decision["stage"],
                "routing_reason": decision["reason"],
                "paper_taxa": paper_taxa,
                "n_relevant_sentences": n_summary,
            }
        )

    except Exception as e:
        print(f"[LLM] ERROR in /query: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# ────────────────────── Feedback endpoint ──────────────────────


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Store user feedback for later analysis / RLHF-style fine-tuning.

    Expected JSON body:
        {
            "query": "...",
            "response": "...",
            "rating": int (e.g. 1-5 or -1/1),
            "comment": "optional free text"
        }
    """
    data = request.get_json(silent=True) or {}
    feedback_dir = LLM_ROOT / "backend" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_file = feedback_dir / "feedback.jsonl"

    import json

    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    print(f"[LLM] Starting LLM backend on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
