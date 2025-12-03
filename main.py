"""
EnViT5 Translation Service

Run locally (example):
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Then open:
    http://localhost:8000/docs
to see the auto-generated Swagger UI.
"""

from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------------------------------------------------------
# Model setup
# -------------------------------------------------------------------

MODEL_NAME = "VietAI/envit5-translation"

# Decide where to run the model: GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model once when the server starts.
# This can take a while on first run because it downloads from Hugging Face.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(
    title="EnViT5 Translation Service",
    description=(
        "Simple HTTP API wrapper around the **VietAI/envit5-translation** model.\n\n"
        "Supports English ↔ Vietnamese translation via a single `/translate` endpoint.\n"
        "- Use `source_lang = 'vi'` when the **input** is Vietnamese (output will be English).\n"
        "- Use `source_lang = 'en'` when the **input** is English (output will be Vietnamese).\n\n"
        "Swagger UI here is designed so non-developers can test the API easily."
    ),
    version="1.0.0",
)


# -------------------------------------------------------------------
# Pydantic models & enums
# -------------------------------------------------------------------

class LanguageEnum(str, Enum):
    """Allowed source languages for the translation API."""
    en = "en"
    vi = "vi"


class TranslateRequest(BaseModel):
    """
    Request body for the /translate endpoint.
    """
    text: str = Field(
        ...,
        description="Input text to translate. Use plain text (no HTML).",
        example="Hôm nay trời đẹp quá"
    )
    source_lang: LanguageEnum = Field(
        ...,
        description=(
            "Language of the **input** text.\n"
            "- `vi` → Vietnamese input, English output\n"
            "- `en` → English input, Vietnamese output"
        ),
        example="vi",
    )
    max_length: int | None = Field(
        256,
        ge=1,
        le=512,
        description=(
            "Maximum length of the generated translation **in tokens**. "
            "Higher values allow longer outputs but are slower. "
            "Default is 256."
        ),
        example=256,
    )


class TranslateResponse(BaseModel):
    """
    Standard response from the /translate endpoint.
    """
    translated_text: str = Field(
        ...,
        description=(
            "Cleaned translation text, without the `en:` or `vi:` prefix. "
            "Use this field in your application."
        ),
        example="The weather is so nice today",
    )
    raw_output: str = Field(
        ...,
        description=(
            "Raw output string directly from the model decoder. "
            "Usually starts with a language prefix like `en:` or `vi:`. "
            "Useful for debugging."
        ),
        example="en: The weather is so nice today",
    )


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def _build_model_input(text: str, source_lang: LanguageEnum) -> str:
    """
    Build the input string expected by the EnViT5 model.

    The model is trained on prompts like:
        - 'vi: Xin chào'  (Vietnamese input)
        - 'en: Hello'     (English input)
    """
    return f"{source_lang.value}: {text}"


def _strip_lang_prefix(output: str) -> str:
    """
    Remove 'en:' or 'vi:' prefix from the model output if present.
    """
    for prefix in ("en:", "vi:"):
        if output.startswith(prefix):
            return output[len(prefix):].strip()
    return output.strip()


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@app.get(
    "/health",
    summary="Health check",
    description=(
        "Quick health-check endpoint.\n\n"
        "Returns a simple JSON payload indicating the service status and whether "
        "the underlying model is currently running on CPU or GPU."
    ),
    tags=["system"],
)
def health_check():
    """
    Health-check endpoint.

    - Use this for monitoring / readiness checks.
    - Example response:
        `{ "status": "ok", "device": "cpu" }`
    """
    return {
        "status": "ok",
        "device": device,
        "model_name": MODEL_NAME,
    }


@app.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate text between English and Vietnamese",
    description=(
        "Translate text using the `VietAI/envit5-translation` model.\n\n"
        "Usage rules:\n"
        "- Set `source_lang = 'vi'` if your **input** is Vietnamese → output will be English.\n"
        "- Set `source_lang = 'en'` if your **input** is English → output will be Vietnamese.\n\n"
        "Example (Vi → En):\n"
        "```json\n"
        "{\n"
        "  \"text\": \"Hôm nay trời đẹp quá\",\n"
        "  \"source_lang\": \"vi\",\n"
        "  \"max_length\": 256\n"
        "}\n"
        "```"
    ),
    tags=["translation"],
)
def translate(req: TranslateRequest) -> TranslateResponse:
    """
    Translate text between English and Vietnamese.

    ### How it works

    1. The service builds a prefixed prompt for the model, e.g.:
       - `vi: Hôm nay trời đẹp quá` (Vietnamese input)\n
       - `en: The weather is so nice today` (English input)\n
    2. The prompt is tokenized and fed into the EnViT5 translation model.
    3. The generated output is decoded and cleaned:
       - `raw_output` keeps the original string (e.g. `en: ...`)\n
       - `translated_text` removes the leading `en:` / `vi:` prefix for convenience.
    """
    # Build model input string with the appropriate language prefix
    input_text = _build_model_input(req.text, req.source_lang)

    # Tokenize input text
    inputs = tokenizer(
        [input_text],          # batch of size 1
        return_tensors="pt",   # return PyTorch tensors
        padding=True,
        truncation=True,
    ).to(device)

    # Run translation (no gradients needed for inference)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=req.max_length or 256,
                num_beams=4,  # beam search for better quality
            )
    except Exception as e:
        # Convert any model error into a clean HTTP 500 for clients
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {e}",
        )

    # Decode token IDs back to text
    decoded = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]  # we only generated one sequence

    # Remove "en:" / "vi:" prefix for the user-facing field
    cleaned = _strip_lang_prefix(decoded)

    return TranslateResponse(
        translated_text=cleaned,
        raw_output=decoded,
    )
