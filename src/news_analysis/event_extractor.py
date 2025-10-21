"""Structured event extraction leveraging an LLM or deterministic fallback."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_interface import LLMClient


EVENT_PROMPT = (
    "Extraia um evento estruturado da notícia a seguir. Retorne um JSON com o "
    "formato {\"evento_tipo\": str, \"metricas\": [ {\"metrica\": str, \"valor\": float opcional, "
    "\"expectativa\": float opcional, \"resultado\": str opcional } ], \"sentimento_geral\": str }."
    "\n\nNotícia:\n{news}"
)


@dataclass
class MetricResult:
    metric: str
    value: Optional[float] = None
    expectation: Optional[float] = None
    outcome: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"metrica": self.metric}
        if self.value is not None:
            data["valor"] = self.value
        if self.expectation is not None:
            data["expectativa"] = self.expectation
        if self.outcome:
            data["resultado"] = self.outcome
        return data


@dataclass
class StructuredEvent:
    event_type: str
    metrics: List[MetricResult] = field(default_factory=list)
    overall_sentiment: str = "neutro"
    impact_rating: Optional[int] = None
    impact_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evento_tipo": self.event_type,
            "metricas": [metric.to_dict() for metric in self.metrics],
            "sentimento_geral": self.overall_sentiment,
            "impacto_potencial": self.impact_rating,
            "justificativa_impacto": self.impact_rationale,
        }


@dataclass
class StructuredEventExtractor:
    llm_client: Optional[LLMClient] = None

    def extract(self, news: str) -> StructuredEvent:
        raw_event = None
        if self.llm_client is not None:
            prompt = EVENT_PROMPT.format(news=news.strip())
            raw_event = self.llm_client.generate(prompt)

        if raw_event:
            try:
                payload = json.loads(raw_event)
                return self._from_json(payload, news)
            except json.JSONDecodeError:
                pass

        return self._rule_based_parse(news)

    # ------------------------------------------------------------------
    def _from_json(self, payload: Dict[str, Any], news: str) -> StructuredEvent:
        metrics = [
            MetricResult(
                metric=metric.get("metrica", ""),
                value=metric.get("valor"),
                expectation=metric.get("expectativa"),
                outcome=metric.get("resultado"),
            )
            for metric in payload.get("metricas", [])
        ]

        event = StructuredEvent(
            event_type=payload.get("evento_tipo", "unknown"),
            metrics=metrics,
            overall_sentiment=payload.get("sentimento_geral", "neutro"),
        )
        event.impact_rating, event.impact_rationale = self._estimate_impact(news, event)
        return event

    # ------------------------------------------------------------------
    def _rule_based_parse(self, news: str) -> StructuredEvent:
        news_lower = news.lower()
        metrics: List[MetricResult] = []

        if "eps" in news_lower or "lucro por ação" in news_lower:
            eps_match = re.search(r"\$?(\d+\.\d+)", news)
            expectation_match = re.search(r"\$?(\d+\.\d+)\s*(?:esperado|expectativa)", news_lower)
            value = float(eps_match.group(1)) if eps_match else None
            expectation = float(expectation_match.group(1)) if expectation_match else None
            outcome = None
            if value and expectation:
                if value > expectation:
                    outcome = "beat"
                elif value < expectation:
                    outcome = "miss"
                else:
                    outcome = "in-line"
            metrics.append(MetricResult("EPS", value=value, expectation=expectation, outcome=outcome))

        revenue_keywords = ["receita", "revenue", "vendas"]
        if any(keyword in news_lower for keyword in revenue_keywords):
            miss_match = re.search(r"(decepcion[aá]o|abaixo|miss)", news_lower)
            beat_match = re.search(r"acima|superou|beat", news_lower)
            outcome = "miss" if miss_match else "beat" if beat_match else None
            metrics.append(MetricResult("revenue", outcome=outcome))

        sentiment = "positivo" if any("super" in news_lower for _ in [0]) else "neutro"
        if "miss" in news_lower or "queda" in news_lower:
            sentiment = "negativo"

        event_type = "earnings_report" if "lucro" in news_lower or "earnings" in news_lower else "corporate_event"
        event = StructuredEvent(event_type=event_type, metrics=metrics, overall_sentiment=sentiment)
        event.impact_rating, event.impact_rationale = self._estimate_impact(news, event)
        return event

    # ------------------------------------------------------------------
    def _estimate_impact(self, news: str, event: StructuredEvent) -> tuple[int, str]:
        """Approximate the impact rating following a chain-of-thought style."""
        impact_score = 3
        rationale_parts: List[str] = []

        lowered = news.lower()
        if any(keyword in lowered for keyword in ("atraso", "delay", "recall", "demissão")):
            impact_score += 3
            rationale_parts.append("Evento operacional crítico identificado.")
        if any(metric.outcome == "beat" for metric in event.metrics):
            impact_score += 2
            rationale_parts.append("Resultados acima do esperado.")
        if any(metric.outcome == "miss" for metric in event.metrics):
            impact_score += 1
            rationale_parts.append("Alguns indicadores abaixo das expectativas.")
        if event.overall_sentiment == "negativo":
            impact_score += 1
            rationale_parts.append("Sentimento negativo predominante.")
        if "adquire" in lowered or "merger" in lowered or "aquisição" in lowered:
            impact_score += 2
            rationale_parts.append("Transação corporativa detectada.")

        impact_score = max(1, min(10, impact_score))
        if not rationale_parts:
            rationale_parts.append("Impacto moderado com informações limitadas.")
        rationale = " ".join(rationale_parts)
        return impact_score, rationale
