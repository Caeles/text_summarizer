import os
import subprocess
import sys
from functools import lru_cache

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

from src.textSummarizer.services.langchain_summarizer import LangChainSummarizer


load_dotenv()


def _clean_env(value, default):
    if value is None:
        return default
    return value.strip().strip('"').strip("'")


def _to_bool(value, default):
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _allowed_origins():
    origins = _clean_env(os.getenv("ALLOWED_ORIGINS"), "http://localhost:8501,http://127.0.0.1:8501")
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


enable_docs = _to_bool(os.getenv("ENABLE_DOCS"), True)
api_port = int(_clean_env(os.getenv("PORT"), "8080"))

app = FastAPI(
    title="TextSummarizer API",
    docs_url="/docs" if enable_docs else None,
    redoc_url="/redoc" if enable_docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_langchain_summarizer():
    model_name = _clean_env(os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_API_MODEL"), "gpt-4o-mini")
    temperature = float(_clean_env(os.getenv("OPENAI_TEMPERATURE"), "0.2"))
    return LangChainSummarizer(model_name=model_name, temperature=temperature)


class LangChainSummarizeRequest(BaseModel):
    text: str
    style: str = Field(default="executive", description="executive | bullets | concise | technical")
    audience: str = Field(default="recruiter", description="target audience of summary")
    language: str = Field(default="fr", description="output language")
    max_words: int = Field(default=180, ge=60, le=500)
    redact_pii: bool = True
    run_quality_check: bool = True


class PredictRequest(BaseModel):
    text: str


@app.get("/", tags=["authentication"])
async def index():
    if enable_docs:
        return RedirectResponse(url="/docs")
    return {"status": "ok", "message": "TextSummarizer API running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/train")
async def training(x_train_token: str = Header(default=None)):
    expected_token = _clean_env(os.getenv("TRAIN_ENDPOINT_TOKEN"), "")

    if expected_token and x_train_token != expected_token:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        return {"message": "Training successful"}

    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Training failed with exit code {exc.returncode}")
    



@app.post("/predict")
async def predict_route(request: PredictRequest):
    from src.textSummarizer.pipeline.predicition_pipeline import PredictionPipeline

    obj = PredictionPipeline()
    text = obj.predict(request.text)
    return {"summary": text}


@app.post("/summarize/langchain")
async def summarize_langchain(request: LangChainSummarizeRequest):
    try:
        result = get_langchain_summarizer().summarize(
            text=request.text,
            style=request.style,
            audience=request.audience,
            language=request.language.lower(),
            max_words=request.max_words,
            redact_pii=request.redact_pii,
            run_quality_check=request.run_quality_check,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain summarization error: {e}")
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=api_port)
