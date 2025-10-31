"""Utilities for validating structured news outputs against market data."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency guidance
    pd = None  # type: ignore

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


_STOOQ_SUFFIXES = {
    "US": ".us",
    "BR": ".sa",
    "CA": ".ca",
    "MX": ".mx",
    "GB": ".uk",
    "CH": ".ch",
    "HK": ".hk",
}


_TICKERS_TO_SKIP = {"F"}


def _attempt_stooq(
    ticker: str,
    start_date: date,
    end_date: date,
    country_code: Optional[str],
):
    """Fetch prices from Stooq as the canonical market data source."""

    if pd is None:  # pragma: no cover - dependency guidance
        return None

    suffix = _STOOQ_SUFFIXES.get((country_code or "US").upper())
    symbol = ticker.lower()
    if suffix and not symbol.endswith(suffix):
        symbol = f"{symbol}{suffix}"

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    LOGGER.info("Buscando preços via Stooq para %s usando símbolo %s", ticker, symbol)

    try:  # pragma: no cover - network variability
        with urlopen(url, timeout=10) as response:
            if response.status != 200:
                LOGGER.debug("Resposta %s da Stooq para %s", response.status, ticker)
                return None
            payload = response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError) as exc:  # pragma: no cover
        LOGGER.debug("Falha ao acessar Stooq para %s: %s", ticker, exc, exc_info=True)
        return None

    if not payload:
        return None

    try:
        df = pd.read_csv(io.StringIO(payload))
    except Exception as exc:  # pragma: no cover - CSV parsing variability
        LOGGER.debug("Falha ao ler CSV da Stooq para %s: %s", ticker, exc, exc_info=True)
        return None

    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if df.empty:
        return None

    df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]
    if df.empty:
        return None

    series = df.set_index("Date")["Close"].astype(float)
    return series.sort_index()


def _normalize_to_bdr(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        return cleaned
    if not cleaned.endswith("34"):
        cleaned = f"{cleaned}34"
    return cleaned


def _fetch_price_series(
    ticker: str,
    start: datetime,
    end: datetime,
    country: Optional[str],
    event_datetime: Optional[datetime] = None,
) -> pd.Series:
    today = datetime.utcnow().date()
    start_date = start.date()
    end_date = end.date()

    if start_date > today:
        raise ValueError(
            "Data do evento no futuro: janela inicia em %s, data atual é %s"
            % (start_date, today)
        )

    # Evita solicitar datas que ainda não ocorreram para reduzir falsos positivos
    # de tickers inválidos.
    effective_end_date = min(end_date, today)

    event_date_str = (
        event_datetime.date().isoformat() if event_datetime is not None else "N/A"
    )

    normalized_ticker = _normalize_to_bdr(ticker)
    country_code = "BR"

    LOGGER.info(
        "Buscando preços para %s (normalizado %s) na data %s via Stooq",
        ticker,
        normalized_ticker,
        event_date_str,
    )

    stooq_series = _attempt_stooq(
        normalized_ticker,
        start_date,
        effective_end_date,
        country_code,
    )
    if stooq_series is not None and not stooq_series.empty:
        LOGGER.info("Dados obtidos via Stooq para %s", normalized_ticker)
        return stooq_series

    raise ValueError(
        "Sem dados de preço retornados via Stooq para %s normalizado como %s"
        % (ticker, normalized_ticker)
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

    today = datetime.utcnow().date()

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

            if event_dt.date() > today:
                LOGGER.debug(
                    "Evento %s (%s) ocorre no futuro em relação à data atual %s; ignorando",
                    raw.get("id"),
                    event_dt.date(),
                    today,
                )
                continue

            if ticker in _TICKERS_TO_SKIP:
                LOGGER.info(
                    "Ignorando ticker %s na data %s conforme configuração de skip.",
                    ticker,
                    event_dt.date(),
                )
                continue
            window_start, window_end = _trading_window(event_dt, max(horizons))

            try:
                price_series = _fetch_price_series(
                    ticker,
                    window_start,
                    window_end,
                    raw.get("country"),
                    event_dt,
                )
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

