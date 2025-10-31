"""Local heuristic-based language model emulation.

Este módulo implementa uma versão extremamente leve de um "LLM local"
baseada em regras heurísticas. Embora não seja um modelo generativo
treinado em larga escala, ele encapsula a lógica necessária para
classificar notícias financeiras e extrair eventos estruturados sem
dependências externas – requisito importante em ambientes isolados.

O objetivo é reproduzir comportamentos típicos de um LLM para o fluxo
de trabalho solicitado: filtrar notícias por relevância e converter o
texto em uma representação JSON rica em informações.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional


# Palavras-chave que direcionam a classificação em diferentes categorias.
MARKET_MOVING_KEYWORDS = {
    "lucro",
    "prejuízo",
    "prejuizo",
    "resultado",
    "balanço",
    "balanco",
    "guidance",
    "projeção",
    "projecao",
    "demissão",
    "demissao",
    "processo",
    "investigação",
    "investigacao",
    "produto",
    "lançamento",
    "lancamento",
    "ipo",
    "fusao",
    "aquisição",
    "aquisicao",
    "paralisação",
    "paralisacao",
    "fábrica",
    "fabrica",
    "profit",
    "revenue",
    "earnings",
    "forecast",
    "guidance",
    "layoff",
    "job cut",
    "lawsuit",
    "investigation",
    "product",
    "launch",
    "ipo",
    "merger",
    "acquisition",
    "strike",
    "factory",
    "delay",
    "supply",
    "vote",
    "shareholder",
    "compensation",
}

FLUFF_KEYWORDS = {
    "doação",
    "doacao",
    "evento",
    "patrocínio",
    "patrocinio",
    "campanha",
    "marketing",
    "entrevista",
    "prêmio",
    "premio",
    "responsabilidade social",
    "donation",
    "sponsorship",
    "marketing",
    "campaign",
    "interview",
    "award",
    "csr",
}

PRODUCT_KEYWORDS = {
    "lançamento",
    "lancamento",
    "produto",
    "serviço",
    "servico",
    "modelo",
    "dispositivo",
    "aplicativo",
    "software",
    "launch",
    "product",
    "service",
    "model",
    "device",
    "app",
    "software",
}

LEGAL_KEYWORDS = {
    "processo",
    "investigação",
    "investigacao",
    "ação coletiva",
    "acao coletiva",
    "litígio",
    "litigio",
    "acusação",
    "acusacao",
    "tribunal",
    "lawsuit",
    "investigation",
    "class action",
    "litigation",
    "accusation",
    "court",
    "settlement",
    "probe",
    "regulator",
}

ANALYST_KEYWORDS = {
    "recomendação",
    "recomendacao",
    "compra",
    "venda",
    "neutra",
    "neutro",
    "preço-alvo",
    "preco-alvo",
    "price target",
    "rating",
    "classificação",
    "classificacao",
    "upgrade",
    "downgrade",
    "avaliação",
    "avaliacao",
    "cobertura",
    "coverage",
    "recommendation",
    "buy",
    "sell",
    "hold",
    "overweight",
    "underweight",
    "initiates",
    "maintains",
    "neutral",
}

HUMAN_RESOURCES_KEYWORDS = {
    "demissão",
    "demissao",
    "corte",
    "reestruturação",
    "reestruturacao",
    "contratação",
    "contratacao",
    "executivo",
    "ceo",
    "cfo",
    "presidente",
    "layoff",
    "job cut",
    "restructuring",
    "hiring",
    "executive",
    "chief",
    "chairman",
}

GOVERNANCE_KEYWORDS = {
    "vote",
    "voto",
    "assembleia",
    "shareholder",
    "proxy",
    "conselho",
    "board",
    "compensation",
    "remuneração",
    "remuneracao",
    "pay package",
    "governança",
    "governanca",
    "proposal",
    "say-on-pay",
}


logger = logging.getLogger(__name__)

EVENT_TYPE_LABELS = {
    "earnings_report": "Resultado financeiro / guidance",
    "product_update": "Atualização de produto ou serviço",
    "management_or_hr": "Mudanças em gestão ou estrutura de pessoal",
    "legal_or_regulatory": "Tema legal ou regulatório",
    "analyst_rating": "Avaliação ou rating de analista",
    "governance": "Deliberação de acionistas ou governança",
}

EARNINGS_KEYWORDS = {
    "lucro",
    "receita",
    "balanço",
    "balanco",
    "ganho",
    "perda",
    "eps",
    "guidance",
    "projeção",
    "projecao",
    "revenue",
    "profit",
    "earnings",
    "forecast",
    "outlook",
}

NEGATIVE_SENTIMENT = {
    "queda",
    "cai",
    "cair",
    "redução",
    "reducao",
    "perda",
    "recuo",
    "recua",
    "baixa",
    "desaceleração",
    "desaceleracao",
    "atraso",
    "multado",
    "investigado",
    "processado",
    "demite",
    "demissão",
    "demissao",
    "contra",
    "against",
    "downgrade",
    "venda",
    "decline",
    "drop",
    "fell",
    "fall",
    "miss",
    "misses",
    "cut",
    "slump",
    "delay",
    "reject",
    "against",
}

POSITIVE_SENTIMENT = {
    "alta",
    "subida",
    "cresce",
    "crescimento",
    "recorde",
    "supera",
    "acima",
    "expande",
    "expansão",
    "expansao",
    "aprova",
    "aprovado",
    "ganha",
    "avanço",
    "avanco",
    "contrata",
    "a favor",
    "approve",
    "upgrade",
    "compra",
    "rally",
    "beat",
    "surpass",
    "gains",
    "growth",
    "improve",
    "supports",
    "for",
}


NUMERIC_PATTERN = re.compile(
    r"(?P<sign>[-+])?\$?(?P<number>\d+[\d.,]*)\s*(?P<unit>milhões|bilhões|biliões|milhões|milhoes|billion|million|%|por cento|percent|usd|us\$|dólares|dolares|dollars|reais|r\$)?",
    re.IGNORECASE,
)


METRIC_LABELS = {
    "eps": "EPS",
    "earnings per share": "EPS",
    "lucro por aç": "EPS",
    "lucro": "profit",
    "profit": "profit",
    "receita": "revenue",
    "revenue": "revenue",
    "sales": "sales",
    "vendas": "sales",
    "margem": "margin",
    "margin": "margin",
    "guidance": "guidance",
    "outlook": "guidance",
    "preço-alvo": "price_target",
    "preco-alvo": "price_target",
    "price target": "price_target",
    "target": "price_target",
}


def normalize_number(raw: str, unit: Optional[str]) -> float:
    """Converte representações textuais de números em float.

    Ex.: "1,5" -> 1.5; "1.500" -> 1500.0. Suporta sufixos de escala como
    milhões/bilhões para gerar valores mais expressivos.
    """

    cleaned = raw.replace(".", "").replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return float("nan")

    unit_normalized = (unit or "").lower()
    if "bilh" in unit_normalized or "bili" in unit_normalized:
        value *= 1_000_000_000
    elif "milh" in unit_normalized:
        value *= 1_000_000
    return value


def detect_metric_name(context: str) -> Optional[str]:
    lower = context.lower()
    for key, name in METRIC_LABELS.items():
        if key in lower:
            return name
    return None


@lru_cache(maxsize=None)
def _compile_keyword(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(_compile_keyword(keyword).search(text) for keyword in keywords)


def detect_event_type(text: str) -> str:
    lower = text.lower()
    if contains_keyword(lower, EARNINGS_KEYWORDS):
        return "earnings_report"
    if contains_keyword(lower, PRODUCT_KEYWORDS):
        return "product_update"
    if contains_keyword(lower, ANALYST_KEYWORDS):
        return "analyst_rating"
    if contains_keyword(lower, LEGAL_KEYWORDS):
        return "legal_or_regulatory"
    if contains_keyword(lower, GOVERNANCE_KEYWORDS):
        return "governance"
    if contains_keyword(lower, HUMAN_RESOURCES_KEYWORDS):
        return "management_or_hr"
    return "general_business"


def detect_category(text: str, *, event_type: Optional[str] = None) -> str:
    """Classifica o texto por relevância considerando o tipo de evento."""

    lower = text.lower()
    if contains_keyword(lower, MARKET_MOVING_KEYWORDS):
        return "Market-Moving"
    if contains_keyword(lower, FLUFF_KEYWORDS):
        return "Fluff/Marketing"

    if event_type in {
        "earnings_report",
        "product_update",
        "management_or_hr",
        "legal_or_regulatory",
        "analyst_rating",
        "governance",
    }:
        return "Market-Moving"

    return "Irrelevante"


def detect_sentiment(text: str) -> str:
    lower = text.lower()
    positive = contains_keyword(lower, POSITIVE_SENTIMENT)
    negative = contains_keyword(lower, NEGATIVE_SENTIMENT)
    if positive and negative:
        return "misto"
    if positive:
        return "positivo"
    if negative:
        return "negativo"
    return "neutro"


@dataclass
class StructuredEvent:
    """Representa o evento estruturado inferido pela heurística."""

    categoria: str
    tipo_evento: str
    sentimento: str
    metricas: List[Dict[str, object]] = field(default_factory=list)
    justificativa: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "categoria": self.categoria,
            "evento_tipo": self.tipo_evento,
            "sentimento_geral": self.sentimento,
            "metricas": self.metricas,
            "justificativa": self.justificativa,
        }


class LocalLLM:
    """Simula um LLM executado localmente via heurísticas determinísticas."""

    def __init__(self, *, min_confidence_threshold: float = 0.3) -> None:
        self.min_confidence_threshold = min_confidence_threshold

    def _extract_metrics(self, text: str) -> List[Dict[str, object]]:
        metrics: List[Dict[str, object]] = []
        for match in NUMERIC_PATTERN.finditer(text):
            number = match.group("number")
            unit = match.group("unit")
            start, end = match.span()
            context_slice = text[max(0, start - 40) : min(len(text), end + 40)]
            metric_name = detect_metric_name(context_slice)
            if metric_name is None:
                continue
            value = normalize_number(number, unit)
            metrics.append(
                {
                    "metrica": metric_name,
                    "valor": value,
                    "texto_original": match.group(0),
                    "unidade": unit,
                }
            )
        return metrics

    def classify(self, text: str) -> StructuredEvent:
        tipo_evento = detect_event_type(text)
        categoria = detect_category(text, event_type=tipo_evento)
        sentimento = detect_sentiment(text)
        metricas = self._extract_metrics(text)

        lower = text.lower()
        market_keyword_hit = contains_keyword(lower, MARKET_MOVING_KEYWORDS)
        fluff_keyword_hit = contains_keyword(lower, FLUFF_KEYWORDS)

        logger.debug(
            "Resultado parcial da classificação: %s",
            {
                "categoria": categoria,
                "tipo_evento": tipo_evento,
                "sentimento": sentimento,
                "metricas": len(metricas),
                "market_keyword_hit": market_keyword_hit,
                "fluff_keyword_hit": fluff_keyword_hit,
            },
        )

        justificativa_parts: List[str] = []
        if categoria == "Market-Moving":
            if market_keyword_hit:
                justificativa_parts.append("Contém termos de impacto direto no negócio")
            else:
                justificativa_parts.append(
                    EVENT_TYPE_LABELS.get(
                        tipo_evento,
                        "Tipo de evento sugere impacto relevante",
                    )
                )
        elif categoria == "Fluff/Marketing":
            if fluff_keyword_hit:
                justificativa_parts.append("Assunto relacionado a marketing ou branding")
            else:
                justificativa_parts.append("Classificado como marketing pelo contexto geral")
        else:
            logger.debug(
                "Fallback para categoria irrelevante: %s",
                {
                    "tipo_evento": tipo_evento,
                    "market_keyword_hit": market_keyword_hit,
                    "fluff_keyword_hit": fluff_keyword_hit,
                },
            )
            justificativa_parts.append("Ausência de gatilhos relevantes detectados")

        if metricas:
            justificativa_parts.append("Números financeiros identificados no texto")

        justificativa = ". ".join(justificativa_parts)

        return StructuredEvent(
            categoria=categoria,
            tipo_evento=tipo_evento,
            sentimento=sentimento,
            metricas=metricas,
            justificativa=justificativa,
        )

    def classify_batch(self, texts: Iterable[str]) -> List[StructuredEvent]:
        return [self.classify(text) for text in texts]

