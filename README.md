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
- 🐳 **Docker support** for easy deployment.

---

## 2. Quick Start (Docker) 🐳

The easiest way to run this service is using Docker:

```bash
# Build and start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

The service will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

**Note**: On first run, the model (~1.2GB) will be downloaded from Hugging Face. This is cached in a Docker volume for subsequent runs.

---

## 3. Manual Installation (Without Docker)

### 3.1. Requirements

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

### 3.2. Clone / Copy This Project

```bash
git clone <your-repo-url> envit5_translation_service
cd envit5_translation_service
```

### 3.3. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3.4. Run the Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 4. Usage Examples

### 4.1. Using Swagger UI (Easiest)

1. Open http://localhost:8000/docs in your browser
2. Click on `POST /translate`
3. Click "Try it out"
4. Enter your request and click "Execute"

### 4.2. Using cURL

**Vietnamese → English:**
```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hôm nay trời đẹp quá",
    "source_lang": "vi",
    "max_length": 256
  }'
```

**English → Vietnamese:**
```bash
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The weather is so nice today",
    "source_lang": "en",
    "max_length": 256
  }'
```

### 4.3. Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/translate",
    json={
        "text": "Xin chào",
        "source_lang": "vi",
        "max_length": 256
    }
)

result = response.json()
print(result["translated_text"])  # Output: Hello
```

### 4.4. Using JavaScript

```javascript
fetch('http://localhost:8000/translate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Xin chào',
    source_lang: 'vi',
    max_length: 256
  })
})
.then(response => response.json())
.then(data => console.log(data.translated_text));
```

---

## 5. API Reference

### `POST /translate`

Translate text between English and Vietnamese.

**Request Body:**
```json
{
  "text": "string (required)",
  "source_lang": "en | vi (required)",
  "max_length": "integer (optional, default: 256, range: 1-512)"
}
```

**Response:**
```json
{
  "translated_text": "string (cleaned output)",
  "raw_output": "string (raw model output with language prefix)"
}
```

### `GET /health`

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "ok",
  "device": "cpu | cuda",
  "model_name": "VietAI/envit5-translation"
}
```

---

## 6. Docker Details

### 6.1. Build Docker Image Manually

```bash
docker build -t envit5-translation-service .
```

### 6.2. Run Docker Container Manually

```bash
docker run -d \
  --name envit5-translation \
  -p 8000:8000 \
  -v huggingface-cache:/root/.cache/huggingface \
  envit5-translation-service
```

### 6.3. GPU Support (NVIDIA)

To use GPU in Docker, you need:
1. NVIDIA GPU drivers installed
2. NVIDIA Container Toolkit installed
3. Modify `docker-compose.yml` to add GPU support:

```yaml
services:
  translation-service:
    # ... existing config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 7. Deployment

### 7.1. Production Considerations

- **Model Caching**: The Hugging Face model is cached in a Docker volume. Ensure this volume persists across deployments.
- **Memory**: The model requires ~2-3GB RAM. Ensure your server has sufficient memory.
- **GPU**: For production workloads, GPU is highly recommended for faster inference.
- **Scaling**: Use a reverse proxy (nginx) and multiple container instances for load balancing.
- **Monitoring**: Use the `/health` endpoint for health checks.

### 7.2. Environment Variables

Currently, the service doesn't require environment variables, but you can add them in `docker-compose.yml` if needed:

```yaml
environment:
  - LOG_LEVEL=info
  - MAX_WORKERS=4
```

---

## 8. Troubleshooting

### Model Download Issues
If the model fails to download:
- Check internet connection
- Ensure Hugging Face is accessible
- Try downloading manually: `transformers-cli download VietAI/envit5-translation`

### Out of Memory
- Reduce `max_length` in requests
- Use GPU if available
- Increase Docker memory limit

### Slow Performance
- Use GPU for faster inference
- Reduce `num_beams` in `main.py` (line 221)
- Use a smaller model if accuracy requirements allow

---

## 9. License

This project is a wrapper around the `VietAI/envit5-translation` model. Please refer to the [model's license](https://huggingface.co/VietAI/envit5-translation) for usage terms.

---

## 10. Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
