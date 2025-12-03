# EnViT5 Translation Service

Simple HTTP API wrapper around the [`VietAI/envit5-translation`](https://huggingface.co/VietAI/envit5-translation) model.

- Supports **English ↔ Vietnamese** translation.
- Exposes a single main endpoint: `POST /translate`.
- Comes with **Swagger UI** at `/docs` so non-developers can test easily.

---

## 1. Features

- 🔁 **Bi-directional translation**: English → Vietnamese, Vietnamese → English.
- 🚀 **FastAPI** backend with auto-generated Swagger UI.
- 🧠 Uses **Hugging Face Transformers** + **PyTorch** under the hood.
- ⚙️ Automatically runs on **GPU** if available, otherwise falls back to CPU.
- 🩺 Health-check endpoint for monitoring / deployment readiness.

---

## 2. Requirements

- **Python**: 3.10+ (recommended)
- **pip** / **virtualenv** (or `venv`)
- Internet access on first run (to download the model from Hugging Face)
- Optional: **NVIDIA GPU** with CUDA for faster inference

Main Python dependencies:

- `fastapi`
- `uvicorn[standard]`
- `transformers`
- `torch`
- `accelerate`

These are already listed in `requirements.txt`.

---

## 3. Installation

### 3.1. Clone / copy this project

```bash
git clone <your-repo-url> envit5_translation_service
cd envit5_translation_service

Install dependencies:
pip install -r requirements.txt
