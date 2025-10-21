"""Interfaces for interacting with language models within the news pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    """Protocol that any language model client must implement."""

    def generate(self, prompt: str) -> str:
        """Return the model response for the provided prompt."""


@dataclass
class RuleBasedLLMClient:
    """A lightweight stand-in for an LLM client used for local testing.

    The implementation relies on simple keyword heuristics to mimic
    a large language model without requiring network access. It enables
    deterministic unit tests while allowing the rest of the pipeline to
    remain agnostic to the actual LLM backend.
    """

    positive_words: tuple[str, ...] = ("beat", "growth", "expands", "record")
    negative_words: tuple[str, ...] = ("miss", "delay", "lawsuit", "decline")

    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        score = 0
        for word in self.positive_words:
            if word in prompt_lower:
                score += 1
        for word in self.negative_words:
            if word in prompt_lower:
                score -= 1

        if "classifique a notícia" in prompt_lower or "classify the following" in prompt_lower:
            if score <= -1:
                return "Market-Moving"
            if score >= 1:
                return "Market-Moving"
            return "Irrelevante"

        if "extraia um evento" in prompt_lower or "structured event" in prompt_lower:
            sentiment = "positivo" if score > 0 else "negativo" if score < 0 else "neutro"
            payload = {
                "evento_tipo": "generic",
                "metricas": [],
                "sentimento_geral": sentiment,
            }
            return json.dumps(payload, ensure_ascii=False)

        if "impacto potencial" in prompt_lower:
            impact_score = max(1, min(10, score * 2 + 5))
            rationale = "Mudança relevante identificada." if score else "Impacto limitado detectado."
            return f"impacto: {impact_score}\njustificativa: {rationale}"

        return ""
