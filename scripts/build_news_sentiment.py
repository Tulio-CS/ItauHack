"""Gera dataset agregado de sentimento a partir das leituras estruturadas.

O script lê o arquivo ``aapl_structured.jsonl`` gerado pela LLM, agrupa as
notícias por dia e ticker e cria um novo arquivo
``investing_news_structured.jsonl`` com métricas agregadas. Ele também salva uma
planilha CSV com os mesmos dados e registra o processo em ``reports/output/logs``.

Quando o ``master_ml_dataset.parquet`` não puder ser aberto (por exemplo, por
falta de bibliotecas de Parquet), o script cai para um modo de contingência que
usa o ``summary_backtest.csv`` apenas para enriquecer os metadados das empresas.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loader import UnsupportedFormatError, load_records


DATA_DIR = Path("data/output")
REPORT_DIR = Path("reports/output")
LOG_DIR = REPORT_DIR / "logs"
NEWS_JSONL = DATA_DIR / "aapl_structured.jsonl"
AGG_JSONL = DATA_DIR / "investing_news_structured.jsonl"
AGG_CSV = REPORT_DIR / "news_sentiment_daily.csv"
TICKER_SUMMARY_CSV = REPORT_DIR / "news_sentiment_ticker_summary.csv"
MASTER_PATH = Path("master_ml_dataset.parquet")
SUMMARY_BACKTEST = REPORT_DIR / "summary_backtest.csv"

SENTIMENT_MAP = {
    "positivo": 1.0,
    "neutro": 0.0,
    "negativo": -1.0,
}

AGG_SENTIMENT_LABEL = {
    1: "positivo",
    0: "neutro",
    -1: "negativo",
}


@dataclass
class DailySentiment:
    ticker: str
    date: str
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    mean_sentiment: float
    sentiment_label: str
    representative_headline: str
    spread_score: float | None
    sharpe_kf: float | None

    def to_json_record(self) -> Dict[str, object]:
        record = asdict(self)
        return record


class NewsAggregator:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(LOG_DIR / "news_sentiment.log", mode="w", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("news_aggregator")

    def load_master_metadata(self) -> Dict[str, Tuple[float | None, float | None]]:
        self.logger.info("Carregando metadados do master_ml_dataset.parquet")
        try:
            records = load_records([MASTER_PATH])
        except UnsupportedFormatError as exc:
            self.logger.warning(
                "Falha ao ler master_ml_dataset.parquet (%s). Recorrendo ao summary_backtest.csv.",
                exc,
            )
            return self._load_summary_backtest()
        except FileNotFoundError:
            self.logger.warning("Arquivo master_ml_dataset.parquet não encontrado. Usando fallback.")
            return self._load_summary_backtest()

        spread_map: Dict[str, Tuple[float | None, float | None]] = {}
        for row in records:
            ticker = row.get("ticker") or row.get("adr") or row.get("bdr")
            if not ticker:
                continue
            try:
                spread_score = float(row.get("spread_score", ""))
            except (TypeError, ValueError):
                spread_score = None
            try:
                sharpe_kf = float(row.get("sharpe_kf", ""))
            except (TypeError, ValueError):
                sharpe_kf = None
            if ticker not in spread_map:
                spread_map[ticker] = (spread_score, sharpe_kf)
        if not spread_map:
            self.logger.info(
                "Metadados do master vazios. Recuando para summary_backtest.csv"
            )
            return self._load_summary_backtest()
        return spread_map

    def _load_summary_backtest(self) -> Dict[str, Tuple[float | None, float | None]]:
        if not SUMMARY_BACKTEST.exists():
            self.logger.error("summary_backtest.csv indisponível. Não há metadados para enriquecer.")
            return {}
        mapping: Dict[str, Tuple[float | None, float | None]] = {}
        with SUMMARY_BACKTEST.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
            header_index = {name: idx for idx, name in enumerate(header)}
            for line in handle:
                parts = line.strip().split(",")
                if len(parts) != len(header):
                    continue
                adr = parts[header_index.get("adr", -1)] if header_index.get("adr", -1) >= 0 else ""
                sharpe_kf = self._safe_float(parts, header_index.get("sharpe_kf"))
                score_kalman_norm = self._safe_float(parts, header_index.get("score_kalman_norm"))
                if score_kalman_norm is None:
                    score_kalman_norm = self._safe_float(parts, header_index.get("median_ratio_diag"))
                if score_kalman_norm is None:
                    score_kalman_norm = sharpe_kf
                if adr:
                    mapping[adr] = (score_kalman_norm, sharpe_kf)
        return mapping

    @staticmethod
    def _safe_float(parts: List[str], index: int | None) -> float | None:
        if index is None or index < 0 or index >= len(parts):
            return None
        try:
            return float(parts[index])
        except (TypeError, ValueError):
            return None

    def load_news(self) -> Iterable[Dict[str, object]]:
        if not NEWS_JSONL.exists():
            raise FileNotFoundError(f"Arquivo de entrada não encontrado: {NEWS_JSONL}")
        with NEWS_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    self.logger.warning("Linha inválida no JSONL: %s", line[:80])

    def aggregate(self) -> List[DailySentiment]:
        metadata = self.load_master_metadata()
        groups: Dict[Tuple[str, str], Dict[str, object]] = defaultdict(lambda: {
            "scores": [],
            "counts": Counter(),
            "headlines": [],
        })

        for entry in self.load_news():
            raw = entry.get("raw", {})
            structured = entry.get("structured_event", {})
            ticker = str(raw.get("ticker", "")).strip()
            if not ticker:
                continue
            dt_str = str(raw.get("datetime", "")).strip()
            try:
                dt = datetime.fromisoformat(dt_str)
            except ValueError:
                self.logger.warning("Data inválida para ticker %s: %s", ticker, dt_str)
                continue
            date_key = dt.date().isoformat()
            sentiment_label = str(structured.get("sentimento_geral", "neutro")).lower()
            sentiment_value = SENTIMENT_MAP.get(sentiment_label, 0.0)

            group = groups[(ticker, date_key)]
            group["scores"].append(sentiment_value)
            group["counts"][sentiment_label] += 1
            headline = str(raw.get("headline", "")).strip()
            if headline:
                group["headlines"].append((dt, headline))

        aggregated: List[DailySentiment] = []
        for (ticker, date_key), payload in sorted(groups.items()):
            scores: List[float] = payload["scores"]
            counts: Counter = payload["counts"]
            if not scores:
                continue
            mean_sentiment = sum(scores) / len(scores)
            sentiment_bucket = 0
            if mean_sentiment > 0.15:
                sentiment_bucket = 1
            elif mean_sentiment < -0.15:
                sentiment_bucket = -1
            label = AGG_SENTIMENT_LABEL[sentiment_bucket]
            positive_count = counts.get("positivo", 0)
            negative_count = counts.get("negativo", 0)
            neutral_count = counts.get("neutro", 0)
            headlines = sorted(payload["headlines"], key=lambda item: item[0])
            representative_headline = headlines[0][1] if headlines else ""

            spread_score, sharpe_kf = metadata.get(ticker, (None, None))

            aggregated.append(
                DailySentiment(
                    ticker=ticker,
                    date=date_key,
                    news_count=len(scores),
                    positive_count=positive_count,
                    negative_count=negative_count,
                    neutral_count=neutral_count,
                    mean_sentiment=round(mean_sentiment, 6),
                    sentiment_label=label,
                    representative_headline=representative_headline,
                    spread_score=spread_score,
                    sharpe_kf=sharpe_kf,
                )
            )

        return aggregated

    def save_outputs(self, records: List[DailySentiment]) -> None:
        self.logger.info("Salvando %d registros agregados", len(records))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with AGG_JSONL.open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(record.to_json_record(), handle, ensure_ascii=False)
                handle.write("\n")
        with AGG_CSV.open("w", encoding="utf-8", newline="") as handle:
            handle.write(
                "ticker,date,news_count,positive_count,negative_count,neutral_count,mean_sentiment,sentiment_label,spread_score,sharpe_kf,representative_headline\n"
            )
            for record in records:
                row = (
                    record.ticker,
                    record.date,
                    str(record.news_count),
                    str(record.positive_count),
                    str(record.negative_count),
                    str(record.neutral_count),
                    f"{record.mean_sentiment:.6f}",
                    record.sentiment_label,
                    "" if record.spread_score is None else f"{record.spread_score:.6f}",
                    "" if record.sharpe_kf is None else f"{record.sharpe_kf:.6f}",
                    record.representative_headline.replace("\n", " ").replace(",", " "),
                )
                handle.write(",".join(row) + "\n")

        ticker_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "news_count": 0,
            "sentiment_sum": 0.0,
        })
        for record in records:
            totals = ticker_totals[record.ticker]
            totals["news_count"] += record.news_count
            totals["sentiment_sum"] += record.mean_sentiment * record.news_count
        with TICKER_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
            handle.write("ticker,total_news,weighted_sentiment\n")
            for ticker, totals in sorted(ticker_totals.items()):
                weighted_sentiment = 0.0
                if totals["news_count"]:
                    weighted_sentiment = totals["sentiment_sum"] / totals["news_count"]
                handle.write(f"{ticker},{int(totals['news_count'])},{weighted_sentiment:.6f}\n")

    def run(self) -> None:
        self.logger.info("Iniciando agregação de notícias")
        records = self.aggregate()
        if not records:
            self.logger.warning("Nenhum registro agregado gerado")
        self.save_outputs(records)
        self.logger.info("Processo concluído")


if __name__ == "__main__":
    NewsAggregator().run()
