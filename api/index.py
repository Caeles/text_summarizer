import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.textSummarizer.services.langchain_summarizer import LangChainSummarizer


load_dotenv()


def _clean_env(value, default):
	if value is None:
		return default
	return value.strip().strip('"').strip("'")


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
	redact_pii: bool = False
	run_quality_check: bool = False


app = FastAPI(title="TextSummarizer Vercel API")


@app.get("/")
async def root():
	return {"status": "ok", "service": "TextSummarizer Vercel API"}


@app.get("/health")
async def health():
	return {"status": "ok"}


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
