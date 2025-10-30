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
    STRUCTURED_EVENT_RETRY_PROMPT,
)
from .output_parsers import JsonOutputParser, OutputParserError

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
        log_llm_responses: bool = False,
    ) -> None:
        self.llm = llm or LocalLLM()
        self.relevance_threshold = relevance_threshold
        self.log_llm_responses = log_llm_responses
        self._relevance_parser = JsonOutputParser(required_keys=("relevance",))
        self._event_parser = JsonOutputParser(
            required_keys=("evento_tipo", "sentimento_geral", "impacto", "metricas")
        )
        logger.info(
            "NewsAnalyzer inicializado com threshold=%s e modelo=%s",
            relevance_threshold,
            getattr(self.llm, "config", None),
        )

    def classify_relevance(self, text: str) -> str:
        prompts = (RELEVANCE_PROMPT, RELEVANCE_RETRY_PROMPT)
        total_attempts = len(prompts)
        last_response: Optional[str] = None
        payload: Optional[Dict[str, Any]] = None

        for attempt, template in enumerate(prompts, start=1):
            prompt = template.format(news=text, previous_response=last_response or "")
            response = self.llm.generate(prompt).strip()
            last_response = response
            if self.log_llm_responses:
                logger.info(
                    "LLM (relevância, tentativa %s) -> %s",
                    attempt,
                    response,
                )
                print(response)
            logger.debug(
                "Resposta do LLM para relevância (tentativa %s): %s",
                attempt,
                response,
            )

            try:
                payload = self._relevance_parser.parse(response)
            except OutputParserError as exc:
                logger.info(
                    "Falha ao interpretar JSON de relevância (tentativa %s): %s",
                    attempt,
                    exc,
                )
                logger.warning(
                    "Resposta bruta do LLM (relevância, tentativa %s): %s",
                    attempt,
                    response,
                )
                if attempt == total_attempts:
                    payload = _repair_relevance_response(response)
                    logger.info(
                        "Reconstruindo JSON de relevância com heurísticas: %s",
                        payload,
                    )
                else:
                    continue

            raw_label = str(payload.get("relevance", ""))
            label = _canonicalize_relevance_label(raw_label)
            logger.debug(
                "Classificação de relevância (tentativa %s): '%s' -> %s",
                attempt,
                text[:120],
                label,
            )
            if self.log_llm_responses:
                final_payload = {"relevance": label}
                logger.info(
                    "JSON final de relevância (tentativa %s): %s",
                    attempt,
                    final_payload,
                )
                print(json.dumps(final_payload, ensure_ascii=False))
            return label

        logger.warning(
            "Não foi possível obter JSON válido de relevância após %s tentativas; aplicando heurísticas. Última resposta: %s",
            len(prompts),
            last_response,
        )
        return _canonicalize_relevance_label(last_response or "")

    def extract_structured_event(self, text: str) -> StructuredEvent:
        prompts = (
            STRUCTURED_EVENT_PROMPT,
            STRUCTURED_EVENT_RETRY_PROMPT,
        )
        last_response: Optional[str] = None
        payload: Optional[Dict[str, Any]] = None

        for attempt, template in enumerate(prompts, start=1):
            prompt = template.format(news=text, previous_response=last_response or "")
            response = self.llm.generate(prompt).strip()
            last_response = response
            if self.log_llm_responses:
                logger.info(
                    "LLM (evento estruturado, tentativa %s) -> %s",
                    attempt,
                    response,
                )
                print(response)
            logger.debug(
                "Resposta do LLM para evento estruturado (tentativa %s): %s",
                attempt,
                response[:240],
            )

            try:
                payload = dict(self._event_parser.parse(response))
            except OutputParserError as exc:
                logger.info(
                    "Falha ao interpretar JSON de evento estruturado (tentativa %s): %s",
                    attempt,
                    exc,
                )
                logger.warning(
                    "Resposta bruta do LLM (evento estruturado, tentativa %s): %s",
                    attempt,
                    response,
                )
                continue

            break

        if not isinstance(payload, dict):
            logger.error(
                "Não foi possível obter JSON estruturado do LLM; utilizando payload padrão. Última resposta: %s",
                last_response,
            )
            payload = _build_fallback_payload(text=text, response=last_response or "")
        else:
            payload = _normalize_event_payload(payload)

        if self.log_llm_responses and isinstance(payload, dict):
            logger.info("JSON final de evento estruturado: %s", payload)
            print(json.dumps(payload, ensure_ascii=False))

        return StructuredEvent.from_json(payload)

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        logger.info("Processando dataframe com %s notícias", len(df))
        for idx, row in enumerate(df.itertuples(index=False), start=1):
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
            if idx % 50 == 0:
                logger.debug("%s notícias analisadas até agora", idx)
        logger.info("Total de notícias market-moving selecionadas: %s", len(records))
        return pd.DataFrame(records)


def combine_news_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate news dataframes while dropping duplicate rows by id/headline."""

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Concatenando dataframes de notícias: total inicial %s", len(combined))
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset="id")
    elif "headline" in combined.columns:
        combined = combined.drop_duplicates(subset="headline")
    logger.info("Total após remoção de duplicatas: %s", len(combined))
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


def _canonicalize_relevance_label(raw_label: str) -> str:
    """Map raw LLM outputs to the expected relevance categories."""

    if not raw_label:
        return "market_moving"

    cleaned = raw_label.strip().lower()
    if cleaned in _KNOWN_LABELS:
        return cleaned

    first_line = cleaned.splitlines()[0]
    for delimiter in (":", "-", "|", ";"):
        if delimiter in first_line:
            first_line = first_line.split(delimiter, 1)[0]
    first_line = first_line.strip()

    normalized = first_line.replace("/", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z_]+", "", normalized)

    if normalized in _KNOWN_LABELS:
        return normalized
    if normalized in _CANONICAL_LABELS:
        return _CANONICAL_LABELS[normalized]

    for label in _KNOWN_LABELS:
        if label in cleaned:
            return label

    if "market" in cleaned and "moving" in cleaned:
        return "market_moving"
    if "fluff" in cleaned or "marketing" in cleaned:
        return "fluff_marketing"

    return "irrelevant"


def _repair_relevance_response(response: str) -> Dict[str, Any]:
    """Build a minimal JSON payload when the LLM omits braces."""

    label = _canonicalize_relevance_label(response)
    return {"relevance": label}


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


def _normalize_event_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    impacto = payload.get("impacto", {})
    if not isinstance(impacto, dict):
        impacto = {}

    metricas = payload.get("metricas", [])
    if not isinstance(metricas, list):
        metricas = []

    payload.setdefault("evento_tipo", "desconhecido")
    payload.setdefault("sentimento_geral", "neutro")
    impacto.setdefault("nota", 0)
    impacto.setdefault("justificativa", "")

    normalized_metricas: List[Dict[str, Any]] = []
    for item in metricas:
        if isinstance(item, dict):
            normalized_metricas.append(item)

    payload["impacto"] = impacto
    payload["metricas"] = normalized_metricas
    return payload
