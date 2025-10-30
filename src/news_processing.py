"""High level orchestration logic for the news analysis workflow."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .local_llm import LocalLLM
from .news_prompts import (
    RELEVANCE_PROMPT,
    RELEVANCE_RETRY_PROMPT,
    STRUCTURED_EVENT_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class StructuredEvent:
    evento_tipo: str
    sentimento_geral: str
    impacto_nota: float
    impacto_justificativa: str
    metricas: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "StructuredEvent":
        impacto = payload.get("impacto", {})
        return cls(
            evento_tipo=payload.get("evento_tipo", "desconhecido"),
            sentimento_geral=payload.get("sentimento_geral", "neutro"),
            impacto_nota=float(impacto.get("nota", 0) or 0),
            impacto_justificativa=impacto.get("justificativa", ""),
            metricas=payload.get("metricas", []) or [],
        )


class NewsAnalyzer:
    """Pipeline to filter, extract features and prepare training data."""

    def __init__(
        self,
        llm: Optional[LocalLLM] = None,
        relevance_threshold: float = 0.5,
    ) -> None:
        self.llm = llm or LocalLLM()
        self.relevance_threshold = relevance_threshold

    def classify_relevance(self, text: str) -> str:
        prompt = RELEVANCE_PROMPT.format(news=text)
        raw_label = self.llm.generate(prompt)
        label = _canonicalize_relevance_label(raw_label)
        if label is not None:
            return label

        logger.warning(
            "Unrecognized relevance label '%s'. Retrying with constrained prompt.",
            raw_label.strip(),
        )
        retry_prompt = RELEVANCE_RETRY_PROMPT.format(news=text)
        retry_label = self.llm.generate(retry_prompt)
        label = _canonicalize_relevance_label(retry_label)
        if label is not None:
            logger.info(
                "Recovered relevance label '%s' after retry.",
                label,
            )
            return label

        logger.error(
            "Failed to classify relevance after retry. Treating as market_moving.",
            extra={"raw_first_pass": raw_label.strip(), "raw_second_pass": retry_label.strip()},
        )
        return "market_moving"

    def extract_structured_event(self, text: str) -> StructuredEvent:
        prompt = STRUCTURED_EVENT_PROMPT.format(news=text)
        response = self.llm.generate(prompt).strip()
        payload: Optional[Dict[str, Any]] = None

        try:
            json_blob = _extract_json_blob(response)
            if json_blob is None:
                logger.info(
                    "LLM response did not contain JSON; falling back to default payload.",
                    extra={"raw_response": response},
                )
            else:
                payload = json.loads(json_blob)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to decode LLM JSON response; falling back to default payload (%s).",
                exc,
                extra={"raw_response": response},
            )
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception(
                "Unexpected error while extracting structured event; using fallback payload.",
                extra={"raw_response": response},
            )

        if not isinstance(payload, dict):
            payload = _build_fallback_payload(text=text, response=response)

        return StructuredEvent.from_json(payload)

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for row in df.itertuples(index=False):
            text = getattr(row, text_column)
            relevance = self.classify_relevance(text)
            if relevance != "market_moving":
                continue
            event = self.extract_structured_event(text)
            item = row._asdict()
            item.update(
                {
                    "relevance": relevance,
                    "evento_tipo": event.evento_tipo,
                    "sentimento_geral": event.sentimento_geral,
                    "impacto_nota": event.impacto_nota,
                    "impacto_justificativa": event.impacto_justificativa,
                    "metricas": event.metricas,
                }
            )
            records.append(item)
        return pd.DataFrame(records)


def combine_news_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate news dataframes while dropping duplicate rows by id/headline."""

    combined = pd.concat(frames, ignore_index=True)
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset="id")
    elif "headline" in combined.columns:
        combined = combined.drop_duplicates(subset="headline")
    return combined


_KNOWN_LABELS = {"market_moving", "fluff_marketing", "irrelevant"}
_CANONICAL_LABELS = {
    "marketmoving": "market_moving",
    "market moving": "market_moving",
    "market_moving": "market_moving",
    "fluffmarketing": "fluff_marketing",
    "fluff marketing": "fluff_marketing",
    "fluff_marketing": "fluff_marketing",
    "irrelevant": "irrelevant",
}


def _canonicalize_relevance_label(raw_label: str) -> Optional[str]:
    """Map raw LLM outputs to one of the expected relevance categories."""

    if not raw_label:
        return None

    cleaned = raw_label.strip().lower()
    # Prefer direct exact matches first
    if cleaned in _KNOWN_LABELS:
        return cleaned

    first_line = cleaned.splitlines()[0]
    # Strip trailing explanations like "market_moving: ..."
    for delimiter in (":", "-", "|", ";"):
        if delimiter in first_line:
            first_line = first_line.split(delimiter, 1)[0]
    first_line = first_line.strip()

    # Replace separators and remove stray characters
    normalized = first_line.replace("/", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z_]+", "", normalized)

    if normalized in _KNOWN_LABELS:
        return normalized
    if normalized in _CANONICAL_LABELS:
        return _CANONICAL_LABELS[normalized]

    # As a last resort, look for keywords inside the full response
    for label in _KNOWN_LABELS:
        if label in cleaned:
            return label
    if "market" in cleaned and "moving" in cleaned:
        return "market_moving"
    if "fluff" in cleaned or "marketing" in cleaned:
        return "fluff_marketing"
    return None


def _extract_json_blob(response: str) -> Optional[str]:
    """Return the JSON object present in ``response`` if the LLM adds wrappers."""

    if not response:
        return None

    # If the response already looks like a JSON object, return it directly
    stripped = response.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # Attempt to find the first JSON object using a stack to match braces
    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]

    logger.debug("Detected unbalanced JSON braces in LLM response", extra={"raw_response": response})
    return None


def _build_fallback_payload(*, text: str, response: str) -> Dict[str, Any]:
    """Return a minimal payload when the LLM could not produce valid JSON."""

    justificativa = (
        "LLM nao retornou JSON estruturado. Resposta original: "
        f"{response[:240]}"
    )
    return {
        "evento_tipo": "desconhecido",
        "sentimento_geral": "neutro",
        "impacto": {"nota": 1, "justificativa": justificativa},
        "metricas": [
            {
                "metrica": "headline_resumida",
                "valor": text[:240],
            }
        ],
    }
