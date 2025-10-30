"""High level orchestration logic for the news analysis workflow."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .local_llm import LocalLLM
from .news_prompts import RELEVANCE_PROMPT, STRUCTURED_EVENT_PROMPT

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
        label = self.llm.generate(prompt).strip().lower()
        if label not in {"market_moving", "fluff_marketing", "irrelevant"}:
            logger.warning("Unexpected relevance label '%s', defaulting to irrelevant", label)
            return "irrelevant"
        return label

    def extract_structured_event(self, text: str) -> StructuredEvent:
        prompt = STRUCTURED_EVENT_PROMPT.format(news=text)
        response = self.llm.generate(prompt)
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM JSON response, returning fallback: %s", response)
            payload = {
                "evento_tipo": "desconhecido",
                "sentimento_geral": "neutro",
                "impacto": {"nota": 1, "justificativa": "Falha ao analisar"},
                "metricas": [],
            }
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
