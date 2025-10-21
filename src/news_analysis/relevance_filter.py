"""News relevance filter powered by an LLM prompt."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List

from .llm_interface import LLMClient


class NewsCategory(str, Enum):
    MARKET_MOVING = "Market-Moving"
    FLUFF_MARKETING = "Fluff/Marketing"
    IRRELEVANT = "Irrelevante"


RELEVANCE_PROMPT = (
    "Classifique a notícia a seguir em uma das categorias: Market-Moving, "
    "Fluff/Marketing ou Irrelevante. Considere Market-Moving para eventos "
    "que possam impactar o preço da ação (lucros, novos produtos, processos, "
    "demissões, guidance). Fluff/Marketing cobre ações promocionais sem "
    "impacto direto. Irrelevante é para itens que não alteram a visão do "
    "investidor. Responda apenas com o nome da categoria.\n\nNotícia:\n{news}"
)


@dataclass
class NewsRelevanceFilter:
    """Apply the first-stage relevance filter."""

    llm_client: LLMClient

    def classify(self, news: str) -> NewsCategory:
        prompt = RELEVANCE_PROMPT.format(news=news.strip())
        response = self.llm_client.generate(prompt).strip()
        for category in NewsCategory:
            if category.value.lower() in response.lower():
                return category
        return NewsCategory.IRRELEVANT

    def filter_market_moving(self, news_items: Iterable[str]) -> List[str]:
        return [item for item in news_items if self.classify(item) == NewsCategory.MARKET_MOVING]
