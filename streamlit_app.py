import os
import time
from typing import Any, Dict

import requests
import streamlit as st


st.set_page_config(
    page_title="AI Summarization",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css');

        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #8b949e;
            margin-bottom: 1rem;
        }
        .metric-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 0.4rem;
            border: 1px solid rgba(255,255,255,0.18);
        }
        .fa-inline {
            margin-right: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        "<div class='main-title'><i class='fa-solid fa-wand-magic-sparkles fa-inline'></i>AI Summarization Studio</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Advanced summarization with map-reduce strategy and style control.</div>",
        unsafe_allow_html=True,
    )


def call_langchain_api(base_url: str, payload: Dict[str, Any], timeout_sec: int = 120) -> Dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/summarize/langchain",
        json=payload,
        timeout=timeout_sec,
    )
    if not response.ok:
        try:
            details = response.json().get("detail", response.text)
        except Exception:
            details = response.text
        raise RuntimeError(f"API {response.status_code}: {details}")
    return response.json()


def render_metadata(metadata: Dict[str, Any]) -> None:
    st.markdown("### Metadata")
    st.markdown(
        f"<span class='metric-badge'>Strategy: {metadata.get('strategy', '-')}</span>"
        f"<span class='metric-badge'>Chunks: {metadata.get('chunk_count', '-')}</span>"
        f"<span class='metric-badge'>Style: {metadata.get('style', '-')}</span>"
        f"<span class='metric-badge'>Language: {metadata.get('language', '-')}</span>",
        unsafe_allow_html=True,
    )

    pii = metadata.get("pii_redaction_counts", {})
    if pii:
        st.caption(
            f"PII redacted → emails: {pii.get('emails', 0)}, phones: {pii.get('phones', 0)}, card-like: {pii.get('card_like', 0)}"
        )


def main() -> None:
    apply_custom_style()
    render_header()

    with st.sidebar:
        st.markdown("<h3><i class='fa-solid fa-sliders fa-inline'></i>Settings</h3>", unsafe_allow_html=True)
        default_api_url = os.getenv("API_URL")
        api_url = st.text_input("API URL", value=default_api_url)

        style = st.selectbox("Style", ["executive", "concise", "technical", "bullets"], index=0)
        audience = st.text_input("Audience", value="recruiter")
        language = st.selectbox("Output language", ["FR", "EN", "ES", "DE"], index=0)
        max_words = st.slider("Max length (words)", min_value=60, max_value=500, value=180, step=10)

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("### Input")
        default_text = (
            "Our team conducted an AI project for automatic summarization of client meetings. "
            "The results show a 42% gain in document processing time, "
            "but we still have factuality errors on technical segments. "
            "The plan is to add a RAG layer and continuous evaluation before deployment."
        )
        source_text = st.text_area("Text to summarize", value=default_text, height=280)

        st.markdown("<span><i class='fa-solid fa-rocket fa-inline'></i>Run</span>", unsafe_allow_html=True)
        summarize_btn = st.button("Generate summary", use_container_width=True)

    with right:
        st.markdown("### Result")
        result_box = st.empty()
        meta_box = st.container()

    if summarize_btn:
        if not source_text.strip():
            st.error("Add some text before starting generation.")
            return

        payload = {
            "text": source_text,
            "style": style,
            "audience": audience,
            "language": language,
            "max_words": max_words,
            "redact_pii": False,
            "run_quality_check": False,
        }

        start = time.time()
        try:
            with st.spinner("Generating summary..."):
                result = call_langchain_api(api_url, payload)

            elapsed = round(time.time() - start, 2)

            summary_text = result.get("summary", "")
            metadata = result.get("metadata", {})

            result_box.success(summary_text)
            st.caption(f"Generation time: {elapsed}s")

            with meta_box:
                render_metadata(metadata)

            with st.expander("Raw JSON"):
                st.json(result)

        except Exception as exc:
            st.error(f"Error during generation: {exc}")


if __name__ == "__main__":
    main()
