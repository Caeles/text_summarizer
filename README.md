# TextSummarizer

Production-style NLP project for text summarization:
- fine-tuning pipeline (PEGASUS)
- FastAPI backend
- Vercel web frontend (`index.html`)
- LangChain orchestration for long-text summarization

## Description
- End-to-end product mindset: training, inference API, UI demo
- Modern LLM stack: Hugging Face + LangChain + FastAPI + Streamlit
- Engineering quality: environment isolation, tests, secure config template
- Real-world constraints: long-document summarization, style control, API hardening

## Features

- `POST /summarize/langchain` with map-reduce summarization for long inputs
- Output style control (`executive`, `concise`, `technical`, `bullets`)
- Strict summary-length control (summary remains shorter than source)
- Live demo and JSON response inspection on Vercel
- Health endpoint and configurable CORS for deployment

## Runtime target

- Primary runtime target: **Vercel**
- Serverless entrypoint: `api/index.py`
- Routing config: `vercel.json`
- Vercel web UI: `index.html`

## Project structure

- `src/textSummarizer/components`: training pipeline components
- `src/textSummarizer/pipeline`: stage orchestration
- `src/textSummarizer/services/langchain_summarizer.py`: advanced summarization service
- `app.py`: FastAPI API layer
- `index.html`: Web frontend
- `streamlit_app.py`: optional local frontend
- `tests/test_app.py`: API smoke tests

## Local development (optional)

1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Configure environment

```bash
cp .env.example .env
```

Then fill at least:
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_TEMPERATURE` (default: `0.2`)

4) Run API locally (optional)

```bash
python app.py
```

5) Run Streamlit frontend locally (optional)

```bash
streamlit run streamlit_app.py
```

## Vercel routes

- `/`
- `/health` → health check
- `/summarize/langchain` → summarization API
- `/docs` → FastAPI docs

## API endpoints

- `GET /health`: health probe
- `GET /train`: training trigger (token-protected if `TRAIN_ENDPOINT_TOKEN` is set)
- `POST /predict`: baseline summarization pipeline
- `POST /summarize/langchain`: advanced summarization endpoint

Example payload for `POST /summarize/langchain`:

```json
{
  "text": "Your long text here...",
  "style": "executive",
  "audience": "recruiter",
  "language": "FR",
  "max_words": 180,
  "redact_pii": false,
  "run_quality_check": false
}
```




# text_summarizer
