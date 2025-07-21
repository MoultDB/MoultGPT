# 🐛 MoultGPT

**MoultGPT** is a modular NLP & CV pipeline for extracting biologically relevant traits related to moulting in arthropods from scientific literature and images. It combines PDF parsing, sentence summarization, trait extraction using a local LLM, and image classification using custom CNNs, with an interactive React interface.

---

## 🚀 Features

- 🧠 Local inference using Mistral 7B (HF Transformers)
- 📄 PDF parsing with GROBID
- 🔍 Sentence-level summarization using TF-IDF + KMeans
- ✍️ Trait extraction in YAML format
- 🧪 Fine-tuning with LoRA on custom data
- 🖼️ Image-based classification of moulting stages (CNN)
- 🌐 Interactive frontend (React) + backend (Flask)

---

## ⚙️ Environment Setup (Python virtualenv)

```bash
# Clone the repo
git clone https://github.com/your-user/MoultGPT.git
cd MoultGPT

# Create and activate virtual environment
python3 -m venv mistral_env
source mistral_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Running the backend (Flask + LLM + CNN)

```bash
cd backend
python app.py
```

Make sure the following models are available:
- `mistral-7B-Instruct-v0.3` (path configurable in `app.py`)
- CNN weights in `models/`:
  - `effnet_multistage_best.pth` – step 1: living vs exuviae
  - `efficientnet_living_best.pth` – step 2: pre-moult / moulting / post-moult

---

## 🧬 CNN-based Image Classification

MoultGPT integrates a two-step image classification pipeline trained on expert-annotated arthropod moulting observations from iNaturalist:

### 🏗️ Architecture

- Step 1: `MultiTaskEffNet`  
  Classifies **exuviae vs living** (EfficientNet + taxon embedding)
- Step 2: `CustomEffNet`  
  Classifies **pre-moult / moulting / post-moult** (if living)

### 🧪 Biological Data Augmentation

Class-aware augmentation was applied, e.g.:

| Class       | Augmentation Strategy |
|-------------|-----------------------|
| exuviae     | Grayscale, light blur |
| post-moult  | Color jitter (teneral effect) |
| moulting    | Center crop only |
| pre-moult   | No augmentation (fragile state) |

### 🧠 Prediction API

You can send a POST request to:

```
/predict_image
```

With:
- `image` – uploaded `.jpg`
- `taxon_id` – one of:  
  `0 = Arachnida`, `1 = Crustacea`, `2 = Hexapoda`, `3 = Myriapoda`

Response:

```json
{
  "prediction": "post-moult",
  "confidence": 0.93
}
```

---

## 📚 Parsing PDFs with GROBID (CLI mode)

```bash
cd tools/grobid/grobid-0.7.1
./gradlew run
```

---

## 🔁 Running Fine-tuning with LoRA

```bash
python main_generate_dataset.py
```

Outputs are saved in `output/`.

---

## 🖼️ Frontend (React)

```bash
cd frontend
npm install
npm start
```

Includes:
- File upload (.txt or PDF)
- DOI input
- Trait query box (LLM)
- Image + taxon prediction (CNN)
- YAML output and feedback system

---

## 📁 Project Structure

```
MoultGPT/
├── backend/               # Flask server (LLM + CNN)
├── CNN/                   # Training scripts and models for image classification
├── frontend/              # React GUI
├── finetuning/            # LoRA training data + scripts
├── paper_handler/         # PDF parsing and summarization
├── tools/grobid/          # GROBID (PDF to TEI)
├── data/, output/, images/ # Inputs and results
├── models/                # Trained CNN weights
└── requirements.txt
```

---

## 🤝 Citation / Acknowledgements

This tool uses:

- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [GROBID](https://github.com/kermitt2/grobid)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [scikit-learn](https://scikit-learn.org/)
- [PyTorch + TorchVision](https://pytorch.org/)

---

## 📬 Contact

For collaborations, bug reports, or questions:  
**Michele Leone** – [michele.leone@outlook.com]  
Project: [moulting.org](https://www.moulting.org)
