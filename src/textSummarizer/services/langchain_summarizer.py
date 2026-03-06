import json
import re
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainSummarizer:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2200,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _redact_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        patterns = {
            "emails": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "phones": r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b",
            "card_like": r"\b(?:\d[ -]*?){13,16}\b",
        }

        redacted = text
        counters: Dict[str, int] = {}

        for label, pattern in patterns.items():
            matches = re.findall(pattern, redacted)
            counters[label] = len(matches)
            if matches:
                redacted = re.sub(pattern, f"[{label.upper()}_REDACTED]", redacted)

        return redacted, counters

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {}

    def _quality_check(self, source_text: str, summary_text: str) -> Dict[str, Any]:
        quality_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You evaluate summary quality. Return strict JSON only with keys: factuality (1-5), completeness (1-5), conciseness (1-5), risk_flags (array of strings), short_rationale (string).",
                ),
                (
                    "human",
                    "Source:\n{source}\n\nSummary:\n{summary}\n\nReturn only JSON.",
                ),
            ]
        )

        chain = quality_prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"source": source_text, "summary": summary_text})
        data = self._extract_json(raw)

        factuality = int(data.get("factuality", 3))
        completeness = int(data.get("completeness", 3))
        conciseness = int(data.get("conciseness", 3))
        confidence = round((factuality + completeness + conciseness) / 15, 2)

        return {
            "factuality": factuality,
            "completeness": completeness,
            "conciseness": conciseness,
            "confidence": confidence,
            "risk_flags": data.get("risk_flags", []),
            "short_rationale": data.get("short_rationale", "Quality estimated by LLM evaluator."),
        }

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\S+", text.strip()))

    def _truncate_to_words(self, text: str, max_words: int) -> str:
        words = re.findall(r"\S+", text.strip())
        if max_words <= 0:
            return ""
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).strip()

    def summarize(
        self,
        text: str,
        style: str = "executive",
        audience: str = "recruiter",
        language: str = "fr",
        max_words: int = 180,
        redact_pii: bool = True,
        run_quality_check: bool = True,
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("The input text is empty.")

        working_text = text.strip()
        source_word_count = self._word_count(working_text)

        if source_word_count <= 1:
            raise ValueError("Input text is too short to generate a shorter summary.")

        target_word_budget = min(max_words, source_word_count - 1)

        redaction_counts: Dict[str, int] = {"emails": 0, "phones": 0, "card_like": 0}

        if redact_pii:
            working_text, redaction_counts = self._redact_pii(working_text)

        chunks: List[str] = self.text_splitter.split_text(working_text)

        map_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior NLP summarization assistant. Keep factual fidelity and preserve key decisions, metrics, and actions.",
                ),
                (
                    "human",
                    "Summarize this chunk for audience={audience} in language={language} with style={style}. Keep it concise. Max {max_words} words.\n\nChunk:\n{chunk}",
                ),
            ]
        )

        reduce_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You merge partial summaries into one concise, non-redundant final summary. Keep critical facts, remove repetition.",
                ),
                (
                    "human",
                    "Combine the partial summaries below into one final summary for audience={audience} in language={language} with style={style}. The final summary MUST be strictly shorter than the source text and max {max_words} words.\n\nPartials:\n{partials}",
                ),
            ]
        )

        direct_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior NLP summarization assistant. Keep factual fidelity and produce concise output.",
                ),
                (
                    "human",
                    "Summarize for audience={audience} in language={language} with style={style}. The summary MUST be strictly shorter than the source text and max {max_words} words.\n\nText:\n{text}",
                ),
            ]
        )

        parser = StrOutputParser()
        map_chain = map_prompt | self.llm | parser
        reduce_chain = reduce_prompt | self.llm | parser
        direct_chain = direct_prompt | self.llm | parser

        if len(chunks) <= 1:
            summary = direct_chain.invoke(
                {
                    "text": working_text,
                    "audience": audience,
                    "language": language,
                    "style": style,
                    "max_words": target_word_budget,
                }
            )
            strategy = "single_pass"
        else:
            partial_summaries = [
                map_chain.invoke(
                    {
                        "chunk": chunk,
                        "audience": audience,
                        "language": language,
                        "style": style,
                        "max_words": max(20, int(target_word_budget / 2)),
                    }
                )
                for chunk in chunks
            ]
            summary = reduce_chain.invoke(
                {
                    "partials": "\n\n".join(partial_summaries),
                    "audience": audience,
                    "language": language,
                    "style": style,
                    "max_words": target_word_budget,
                }
            )
            strategy = "map_reduce"

        summary = self._truncate_to_words(summary, target_word_budget)
        summary_word_count = self._word_count(summary)

        quality = None
        if run_quality_check:
            quality = self._quality_check(working_text, summary)

        return {
            "summary": summary.strip(),
            "metadata": {
                "strategy": strategy,
                "chunk_count": len(chunks),
                "style": style,
                "audience": audience,
                "language": language,
                "max_words": max_words,
                "target_word_budget": target_word_budget,
                "source_word_count": source_word_count,
                "summary_word_count": summary_word_count,
                "pii_redaction_enabled": redact_pii,
                "pii_redaction_counts": redaction_counts,
            },
            "quality": quality,
        }
