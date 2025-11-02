"""Integra dados de notícias estruturadas com métricas de mercado locais.

O módulo foi desenhado para rodar apenas com bibliotecas da stdlib,
permitindo que o pipeline funcione mesmo em ambientes desconectados.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Iterator, List, Tuple

logger = logging.getLogger(__name__)

SENTIMENT_SCORES: Dict[str, int] = {
    "positivo": 1,
    "otimista": 1,
    "alta": 1,
    "bullish": 1,
    "negativo": -1,
    "pessimista": -1,
    "baixa": -1,
    "bearish": -1,
    "neutro": 0,
    "neutro+": 0,
    "neutro-": 0,
    "misto": 0,
}


@dataclass
class NewsEvent:
    ticker: str
    date: str
    sentiment_label: str
    sentiment_score: float
    source: str
    headline: str


@dataclass
class DailySentiment:
    ticker: str
    date: str
    total_news: int
    positive: int
    negative: int
    neutral: int
    mean_score: float
    aggregate_label: str


@dataclass
class OverallSentiment:
    ticker: str
    total_news: int
    mean_score: float
    aggregate_label: str


def _normalise_datetime(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    try:
        # Handle offsets like "+00:00" or microseconds.
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _sentiment_to_score(label: str | None) -> Tuple[str, float]:
    if not label:
        return "desconhecido", 0.0
    label_norm = label.strip().lower()
    score = SENTIMENT_SCORES.get(label_norm)
    if score is None:
        logger.debug("Sentimento desconhecido: %s", label)
        return label_norm, 0.0
    return label_norm, float(score)


def load_structured_news(path: Path) -> List[NewsEvent]:
    events: List[NewsEvent] = []
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de notícias não encontrado: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Linha ignorada por erro de JSON: %s", exc)
                continue

            raw = payload.get("raw", {})
            structured = payload.get("structured_event", {})

            ticker = raw.get("ticker") or structured.get("ticker") or "UNKNOWN"
            date = _normalise_datetime(raw.get("datetime"))
            if not date:
                logger.debug("Evento sem data utilizável: %s", raw)
                continue

            label, score = _sentiment_to_score(structured.get("sentimento_geral"))
            events.append(
                NewsEvent(
                    ticker=ticker.strip().upper(),
                    date=date,
                    sentiment_label=label,
                    sentiment_score=score,
                    source=str(raw.get("source") or raw.get("publisher") or ""),
                    headline=str(raw.get("headline") or raw.get("title") or ""),
                )
            )
    logger.info("Carregados %s eventos estruturados", len(events))
    return events


def aggregate_daily_sentiment(events: Iterable[NewsEvent]) -> List[DailySentiment]:
    buckets: Dict[Tuple[str, str], List[NewsEvent]] = defaultdict(list)
    for event in events:
        buckets[(event.ticker, event.date)].append(event)

    aggregates: List[DailySentiment] = []
    for (ticker, date), items in sorted(buckets.items()):
        scores = [item.sentiment_score for item in items]
        positives = sum(1 for item in items if item.sentiment_score > 0)
        negatives = sum(1 for item in items if item.sentiment_score < 0)
        neutrals = len(items) - positives - negatives
        avg_score = mean(scores) if scores else 0.0
        if avg_score > 0.15:
            label = "positivo"
        elif avg_score < -0.15:
            label = "negativo"
        elif positives and negatives:
            label = "misto"
        else:
            label = "neutro"
        aggregates.append(
            DailySentiment(
                ticker=ticker,
                date=date,
                total_news=len(items),
                positive=positives,
                negative=negatives,
                neutral=neutrals,
                mean_score=round(avg_score, 4),
                aggregate_label=label,
            )
        )
    logger.info("Gerados %s agregados diários", len(aggregates))
    return aggregates


def aggregate_overall_sentiment(events: Iterable[NewsEvent]) -> List[OverallSentiment]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for event in events:
        buckets[event.ticker].append(event.sentiment_score)

    overall: List[OverallSentiment] = []
    for ticker, scores in sorted(buckets.items()):
        avg_score = mean(scores) if scores else 0.0
        if avg_score > 0.1:
            label = "positivo"
        elif avg_score < -0.1:
            label = "negativo"
        else:
            label = "neutro"
        overall.append(
            OverallSentiment(
                ticker=ticker,
                total_news=len(scores),
                mean_score=round(avg_score, 4),
                aggregate_label=label,
            )
        )
    return overall


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Arquivo salvo: %s", path)


def save_daily_sentiment(path: Path, aggregates: Iterable[DailySentiment]) -> None:
    rows = [
        {
            "ticker": item.ticker,
            "date": item.date,
            "total_news": item.total_news,
            "positive": item.positive,
            "negative": item.negative,
            "neutral": item.neutral,
            "mean_score": f"{item.mean_score:.4f}",
            "aggregate_label": item.aggregate_label,
        }
        for item in aggregates
    ]
    _write_csv(
        path,
        rows,
        ["ticker", "date", "total_news", "positive", "negative", "neutral", "mean_score", "aggregate_label"],
    )


def save_overall_sentiment(path: Path, aggregates: Iterable[OverallSentiment]) -> None:
    rows = [
        {
            "ticker": item.ticker,
            "total_news": item.total_news,
            "mean_score": f"{item.mean_score:.4f}",
            "aggregate_label": item.aggregate_label,
        }
        for item in aggregates
    ]
    _write_csv(path, rows, ["ticker", "total_news", "mean_score", "aggregate_label"])


def load_master_summary(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Resumo mestre não encontrado: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["adr"].strip().upper(): row for row in reader}


def merge_sentiment_with_master(
    daily: Iterable[DailySentiment],
    master_summary: Dict[str, Dict[str, str]],
    output_path: Path,
) -> None:
    rows: List[Dict[str, str]] = []
    for item in daily:
        base = master_summary.get(item.ticker)
        if not base:
            continue
        merged = {
            "ticker": item.ticker,
            "date": item.date,
            "total_news": str(item.total_news),
            "mean_score": f"{item.mean_score:.4f}",
            "aggregate_label": item.aggregate_label,
            "pnl_base": base.get("pnl_base", ""),
            "trades_base": base.get("trades_base", ""),
            "sharpe_base": base.get("sharpe_base", ""),
            "median_ratio_diag": base.get("median_ratio_diag", ""),
        }
        rows.append(merged)

    fieldnames = [
        "ticker",
        "date",
        "total_news",
        "mean_score",
        "aggregate_label",
        "pnl_base",
        "trades_base",
        "sharpe_base",
        "median_ratio_diag",
    ]
    _write_csv(output_path, rows, fieldnames)


def copy_structured_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
    logger.info("Arquivo estruturado copiado para %s", target)
