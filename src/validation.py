"""Utilities for validating structured news outputs against market data."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency guidance
    pd = None  # type: ignore

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - dependency guidance
    yf = None


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class HorizonResult:
    horizon_days: int
    total: int
    correct: int
    incorrect: int
    neutral_hits: int

    @property
    def accuracy(self) -> float:
        if not self.total:
            return float("nan")
        return self.correct / self.total


SENTIMENT_TO_EXPECTED_MOVE = {
    "positivo": 1,
    "positive": 1,
    "negativo": -1,
    "negative": -1,
    "neutro": 0,
    "neutral": 0,
}


def load_structured_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                LOGGER.warning("Linha ignorada por não ser JSON válido: %s", line[:120])
    return records


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Não foi possível converter '{value}' em datetime ISO 8601") from exc


def _trading_window(date: datetime, days_forward: int) -> Tuple[datetime, datetime]:
    start = date - timedelta(days=5)
    end = date + timedelta(days=days_forward + 7)
    return start, end


_ALT_SUFFIXES = (".US", ".SA", ".MX", ".NE", ".TO", ".L", ".SW", ".HK")


def _candidate_symbols(ticker: str) -> Sequence[str]:
    """Generate alternative symbols that yfinance may accept.

    We first yield the original ticker and then append a few common suffixes that
    map to alternative exchanges (ex.: ``.US`` for Stooq listings). This helps
    when Yahoo Finance lacks timezone metadata for the canonical ticker
    (``YFTzMissingError``), which otherwise results in empty downloads.
    """

    normalized = ticker.strip()
    seen = set()
    for symbol in (normalized, normalized.replace(".", "-"), normalized.replace("-", ".")):
        if symbol and symbol not in seen:
            seen.add(symbol)
            yield symbol

    # Fallback to alternate exchanges only if the canonical ticker is short and
    # alphanumeric (e.g., ``F`` -> ``F.US``)
    base = normalized.split(".")[0]
    if base.isalpha() and 1 <= len(base) <= 5:
        for suffix in _ALT_SUFFIXES:
            candidate = f"{base}{suffix}"
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _attempt_download(symbol: str, start_date: date, end_date: date):
    try:
        return yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network variability
        LOGGER.debug("Download primário falhou para %s: %s", symbol, exc, exc_info=True)
        return None


def _attempt_history(symbol: str, start_date: date, end_date: date):
    ticker_client = yf.Ticker(symbol)
    history_kwargs = {
        "start": start_date,
        "end": end_date,
        "auto_adjust": False,
        "actions": False,
    }
    try:
        return ticker_client.history(**history_kwargs, raise_errors=False)
    except TypeError:  # pragma: no cover - older yfinance versions
        try:
            return ticker_client.history(**history_kwargs)
        except Exception as exc:  # pragma: no cover - network variability
            LOGGER.debug("Fallback history falhou para %s: %s", symbol, exc, exc_info=True)
            return None
    except Exception as exc:  # pragma: no cover - network variability
        LOGGER.debug("Fallback history falhou para %s: %s", symbol, exc, exc_info=True)
        return None


def _fetch_price_series(ticker: str, start: datetime, end: datetime) -> pd.Series:
    if yf is None:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'yfinance' é necessário para baixar preços. "
            "Instale-o com 'pip install yfinance' e rode novamente."
        )

    today = datetime.utcnow().date()
    start_date = start.date()
    end_date = end.date()

    if start_date > today:
        raise ValueError(
            "Data do evento no futuro: janela inicia em %s, data atual é %s"
            % (start_date, today)
        )

    # Evita solicitar datas que ainda não ocorreram para reduzir falsos positivos
    # de tickers inválidos. Mantemos um dia adicional na requisição para capturar
    # o pregão seguinte.
    effective_end_date = min(end_date, today)
    request_end_date = effective_end_date + timedelta(days=1)

    attempted_symbols = []
    for symbol in _candidate_symbols(ticker):
        attempted_symbols.append(symbol)
        LOGGER.debug(
            "Baixando preços para %s (tentativa %s) de %s a %s",
            symbol,
            len(attempted_symbols),
            start_date,
            effective_end_date,
        )

        data = _attempt_download(symbol, start_date, request_end_date)
        if data is None or data.empty:
            LOGGER.debug("Tentando fallback com yf.Ticker.history para %s", symbol)
            data = _attempt_history(symbol, start_date, request_end_date)

        if data is not None and not data.empty:
            # Limita as datas ao intervalo efetivo utilizado para requisição.
            data = data.loc[
                (data.index.date >= start_date)
                & (data.index.date <= effective_end_date)
            ]

        if data is None or data.empty:
            continue

        price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
        if price_column not in data.columns:
            LOGGER.debug("Sem coluna de preço em %s (colunas: %s)", symbol, list(data.columns))
            continue

        if symbol != ticker:
            LOGGER.info("Uso do símbolo alternativo %s para baixar preços de %s", symbol, ticker)

        return data[price_column].dropna()

    raise ValueError(
        "Sem dados de preço retornados para %s (tentativas: %s)"
        % (ticker, ", ".join(attempted_symbols))
    )


def _price_on_or_after(series: pd.Series, reference: datetime) -> Optional[Tuple[pd.Timestamp, float]]:
    filtered = series[series.index >= pd.Timestamp(reference.date())]
    if filtered.empty:
        return None
    first_idx = filtered.index[0]
    return first_idx, float(filtered.loc[first_idx])


def _direction_from_return(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def evaluate_predictions(
    records: Sequence[Dict[str, object]],
    horizons: Sequence[int] = (1, 3, 5),
    neutral_threshold: float = 0.01,
) -> Tuple[pd.DataFrame, Dict[int, HorizonResult], pd.DataFrame]:
    if pd is None:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'pandas' é necessário para montar as tabelas de validação. "
            "Instale-o com 'pip install pandas' e rode novamente."
        )
    """Match structured predictions against price direction.

    Returns a tuple with:
        - detailed record dataframe per horizon
        - horizon summary stats
        - aggregate confusion matrix per horizon
    """

    detailed_rows: List[Dict[str, object]] = []
    summaries: Dict[int, HorizonResult] = {}
    confusion_maps: Dict[int, Counter] = defaultdict(Counter)

    grouped_by_ticker: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        raw = record.get("raw", {})
        ticker = str(raw.get("ticker") or "").strip().upper()
        if not ticker:
            LOGGER.debug("Registro sem ticker, ignorando: %s", record)
            continue
        grouped_by_ticker[ticker].append(record)

    for ticker, ticker_records in grouped_by_ticker.items():
        ticker_records.sort(key=lambda r: _parse_datetime(r["raw"]["datetime"]))
        for record in ticker_records:
            raw = record["raw"]
            structured = record.get("structured_event", {})
            sentiment = str(structured.get("sentimento_geral", "")).lower()
            expected_move = SENTIMENT_TO_EXPECTED_MOVE.get(sentiment)
            if expected_move is None:
                LOGGER.debug("Sentimento desconhecido '%s' para registro %s", sentiment, raw.get("id"))
                continue

            event_dt = _parse_datetime(raw["datetime"])
            window_start, window_end = _trading_window(event_dt, max(horizons))

            try:
                price_series = _fetch_price_series(ticker, window_start, window_end)
            except Exception as exc:  # pragma: no cover - network variability
                LOGGER.warning("Falha ao obter preços para %s (%s): %s", ticker, raw.get("id"), exc)
                continue

            base = _price_on_or_after(price_series, event_dt)
            if base is None:
                LOGGER.debug("Sem preço base para %s em %s", ticker, event_dt)
                continue
            base_date, base_price = base

            for horizon in horizons:
                target = _price_on_or_after(price_series, event_dt + timedelta(days=horizon))
                if target is None:
                    LOGGER.debug("Sem preço para %s +%sd", ticker, horizon)
                    continue
                target_date, target_price = target
                ret = (target_price - base_price) / base_price
                realized_dir = _direction_from_return(ret, neutral_threshold)

                is_correct = realized_dir == expected_move
                is_neutral_hit = expected_move == 0 and realized_dir == 0
                confusion_maps[horizon][(expected_move, realized_dir)] += 1

                detailed_rows.append(
                    {
                        "id": raw.get("id"),
                        "ticker": ticker,
                        "headline": raw.get("headline"),
                        "event_datetime": event_dt,
                        "base_date": base_date,
                        "target_date": target_date,
                        "horizon_days": horizon,
                        "sentiment": sentiment,
                        "expected_move": expected_move,
                        "return": ret,
                        "realized_direction": realized_dir,
                        "correct": is_correct,
                        "neutral_hit": is_neutral_hit,
                    }
                )

    detailed_df = pd.DataFrame(detailed_rows)
    if detailed_df.empty:
        LOGGER.warning("Nenhum registro com dados de preço foi avaliado.")
        return detailed_df, summaries, pd.DataFrame()

    for horizon in horizons:
        horizon_df = detailed_df[detailed_df["horizon_days"] == horizon]
        total = len(horizon_df)
        correct = int(horizon_df["correct"].sum())
        neutral_hits = int(horizon_df["neutral_hit"].sum())
        summaries[horizon] = HorizonResult(
            horizon_days=horizon,
            total=total,
            correct=correct,
            incorrect=total - correct,
            neutral_hits=neutral_hits,
        )

    confusion_rows = []
    for horizon, counts in confusion_maps.items():
        for (expected, realized), count in counts.items():
            confusion_rows.append(
                {
                    "horizon_days": horizon,
                    "expected_move": expected,
                    "realized_move": realized,
                    "count": count,
                }
            )
    confusion_df = pd.DataFrame(confusion_rows)

    return detailed_df, summaries, confusion_df


def create_accuracy_table(summaries: Dict[int, HorizonResult]) -> pd.DataFrame:
    if pd is None:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'pandas' é necessário para montar as tabelas de validação. "
            "Instale-o com 'pip install pandas' e rode novamente."
        )
    rows = []
    for horizon, result in sorted(summaries.items()):
        rows.append(
            {
                "horizon_days": horizon,
                "total": result.total,
                "correct": result.correct,
                "incorrect": result.incorrect,
                "neutral_hits": result.neutral_hits,
                "accuracy": result.accuracy,
            }
        )
    return pd.DataFrame(rows)


def generate_accuracy_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'matplotlib' é necessário para gerar gráficos. "
            "Instale-o com 'pip install matplotlib' e rode novamente."
        ) from exc

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["horizon_days"], summary_df["accuracy"], color="#1f77b4")
    plt.ylim(0, 1)
    plt.xlabel("Horizonte (dias)")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por horizonte de validação")
    for x, acc in zip(summary_df["horizon_days"], summary_df["accuracy"]):
        if math.isnan(acc):
            label = "N/A"
        else:
            label = f"{acc:.1%}"
        plt.text(x, min(acc if not math.isnan(acc) else 0.0, 0.95), label, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_confusion_heatmap(confusion_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'matplotlib' é necessário para gerar gráficos. "
            "Instale-o com 'pip install matplotlib' e rode novamente."
        ) from exc

    try:
        import seaborn as sns
    except ImportError:  # pragma: no cover - fallback viz path
        LOGGER.warning(
            "Seaborn não instalado; salvando heatmap simples. Instale com 'pip install seaborn' para gráficos completos."
        )
        pivot = confusion_df.pivot_table(
            index="expected_move",
            columns=["horizon_days", "realized_move"],
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        plt.figure(figsize=(10, 6))
        plt.imshow(pivot, cmap="Blues", aspect="auto")
        plt.title("Mapa de calor das previsões x movimentos realizados")
        plt.xlabel("Horizon / Movimento realizado")
        plt.ylabel("Movimento esperado")
        plt.colorbar(label="Ocorrências")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    pivot = confusion_df.pivot_table(
        index="expected_move",
        columns=["horizon_days", "realized_move"],
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Mapa de calor das previsões x movimentos realizados")
    plt.xlabel("Horizon / Movimento realizado")
    plt.ylabel("Movimento esperado")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

