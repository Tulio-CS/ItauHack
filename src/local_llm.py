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
import re
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
}


NUMERIC_PATTERN = re.compile(
    r"(?P<sign>[-+])?\$?(?P<number>\d+[\d.,]*)\s*(?P<unit>milhões|bilhões|biliões|%|por cento|usd|us\$|dólares|dolares|reais|r\$)?",
    re.IGNORECASE,
)


METRIC_LABELS = {
    "eps": "EPS",
    "lucro por aç": "EPS",
    "lucro": "profit",
    "receita": "revenue",
    "vendas": "sales",
    "margem": "margin",
    "guidance": "guidance",
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


def detect_event_type(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in EARNINGS_KEYWORDS):
        return "earnings_report"
    if any(word in lower for word in PRODUCT_KEYWORDS):
        return "product_update"
    if any(word in lower for word in HUMAN_RESOURCES_KEYWORDS):
        return "management_or_hr"
    if any(word in lower for word in LEGAL_KEYWORDS):
        return "legal_or_regulatory"
    return "general_business"


def detect_category(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in MARKET_MOVING_KEYWORDS):
        return "Market-Moving"
    if any(word in lower for word in FLUFF_KEYWORDS):
        return "Fluff/Marketing"
    return "Irrelevante"


def detect_sentiment(text: str) -> str:
    lower = text.lower()
    positive = any(word in lower for word in POSITIVE_SENTIMENT)
    negative = any(word in lower for word in NEGATIVE_SENTIMENT)
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
        categoria = detect_category(text)
        tipo_evento = detect_event_type(text)
        sentimento = detect_sentiment(text)
        metricas = self._extract_metrics(text)

        justificativa_parts: List[str] = []
        if categoria == "Market-Moving":
            justificativa_parts.append("Contém termos de impacto direto no negócio")
        elif categoria == "Fluff/Marketing":
            justificativa_parts.append("Assunto relacionado a marketing ou branding")
        else:
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

