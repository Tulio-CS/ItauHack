"""Utilities for validating structured news outputs against market data."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency guidance
    pd = None  # type: ignore

LOGGER = logging.getLogger(__name__)


PAIRS_BDR_TO_ADR = {
    "AAPL34": "AAPL",
    "MSFT34": "MSFT",
    "NVDC34": "NVDA",
    "AMZO34": "AMZN",
    "GOGL34": "GOOGL",
    "M1TA34": "META",
    "TSLA34": "TSLA",
    "TSMC34": "TSM",
    "AVGO34": "AVGO",
    "BABA34": "BABA",
    "JDCO34": "JD",
    "BERK34": "BRK.B",
    "LILY34": "LLY",
    "VISA34": "V",
    "JPMC34": "JPM",
    "EXXO34": "XOM",
    "JNJB34": "JNJ",
    "MSCD34": "MA",
    "PGCO34": "PG",
    "COWC34": "COST",
    "BOAC34": "BAC",
    "NFLX34": "NFLX",
    "A1MD34": "AMD",
    "COCA34": "KO",
    "PEPB34": "PEP",
    "WALM34": "WMT",
    "MCDC34": "MCD",
    "DISB34": "DIS",
    "CATP34": "CAT",
    "ITLC34": "INTC",
    "CSCO34": "CSCO",
    "ORCL34": "ORCL",
    "SSFO34": "CRM",
    "ADBE34": "ADBE",
    "NIKE34": "NKE",
    "SBUB34": "SBUX",
    "BOEI34": "BA",
    "GSGI34": "GS",
    "MSBR34": "MS",
    "FDMO34": "F",
    "GMCO34": "GM",
    "PFIZ34": "PFE",
    "CHVX34": "CVX",
    "PYPL34": "PYPL",
    "C2OI34": "COIN",
    "U1BE34": "UBER",
    "A1BN34": "ABNB",
}

_MANUAL_NEWS_ALIASES = {
    "GOOGL": "GOGL34",
    "GOOG": "GOGL34",
    "GOGL35": "GOGL34",
    "GOGL34": "GOGL34",
    "BRK.B": "BERK34",
    "BRKB": "BERK34",
    "BRK-B": "BERK34",
    "BRK B": "BERK34",
    "GOOG34": "GOGL34",
    "CRM": "SSFO34",
    "TSLA": "TSLA34",
    "META": "M1TA34",
    "NVDA": "NVDC34",
    "AMZN": "AMZO34",
    "AAPL": "AAPL34",
    "MSFT": "MSFT34",
    "JD": "JDCO34",
    "TSM": "TSMC34",
    "AVGO": "AVGO34",
    "BABA": "BABA34",
    "LLY": "LILY34",
    "V": "VISA34",
    "JPM": "JPMC34",
    "XOM": "EXXO34",
    "JNJ": "JNJB34",
    "MA": "MSCD34",
    "PG": "PGCO34",
    "COST": "COWC34",
    "BAC": "BOAC34",
    "NFLX": "NFLX34",
    "AMD": "A1MD34",
    "KO": "COCA34",
    "PEP": "PEPB34",
    "WMT": "WALM34",
    "MCD": "MCDC34",
    "DIS": "DISB34",
    "CAT": "CATP34",
    "INTC": "ITLC34",
    "CSCO": "CSCO34",
    "ORCL": "ORCL34",
    "ADBE": "ADBE34",
    "NKE": "NIKE34",
    "SBUX": "SBUB34",
    "BA": "BOEI34",
    "GS": "GSGI34",
    "MS": "MSBR34",
    "F": "FDMO34",
    "GM": "GMCO34",
    "PFE": "PFIZ34",
    "CVX": "CHVX34",
    "PYPL": "PYPL34",
    "COIN": "C2OI34",
    "UBER": "U1BE34",
    "ABNB": "A1BN34",
}

ROLLING_WINDOW_SIZES: Tuple[int, ...] = (1, 3, 5)


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


class PriceFetchError(RuntimeError):
    """Raised when price data could not be recovered for a ticker."""

    def __init__(
        self,
        ticker: str,
        attempted_symbols: Sequence[str],
        message: str,
        *,
        details: Optional[str] = None,
    ) -> None:
        full_message = message
        if details:
            full_message = f"{message} | Detalhes: {details}"
        super().__init__(full_message)
        self.ticker = ticker
        self.attempted_symbols = list(attempted_symbols)
        self.details = details


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


def _ensure_naive_datetime_series(values: Sequence[object] | "pd.Series") -> "pd.Series":
    if pd is None:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'pandas' é necessário para normalizar datas. "
            "Instale-o com 'pip install pandas' e rode novamente."
        )

    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if not isinstance(converted, pd.Series):
        converted = pd.Series(converted)
    return converted.dt.tz_localize(None)


def _ensure_naive_timestamp(value: object) -> "pd.Timestamp":
    series = _ensure_naive_datetime_series([value])
    return series.iloc[0]


def _trading_window(date: datetime, days_forward: int) -> Tuple[datetime, datetime]:
    start = date - timedelta(days=5)
    end = date + timedelta(days=days_forward + 7)
    return start, end


class LocalPriceStore:
    """Carrega séries históricas de preços a partir das planilhas fornecidas."""

    def __init__(self, bdr_path: Path, origem_path: Path) -> None:
        if pd is None:  # pragma: no cover - dependency guidance
            raise RuntimeError(
                "O pacote 'pandas' é necessário para carregar as planilhas de preços. "
                "Instale-o com 'pip install pandas openpyxl'."
            )

        self._bdr_path = bdr_path
        self._origem_path = origem_path
        self._bdr_data = self._load_workbook(bdr_path)
        self._adr_data = self._load_workbook(origem_path)
        self._alias_map = self._build_alias_map()

    @staticmethod
    def _normalize_sheet_name(name: str) -> str:
        return name.strip().upper()

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(name))
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return normalized.strip().lower()

    @staticmethod
    def _normalize_numeric(series: "pd.Series") -> "pd.Series":
        def _convert(value: object) -> Optional[float]:
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none"}:
                return None

            text = text.replace("\u00a0", "").replace(" ", "")
            if "," in text:
                text = text.replace(".", "").replace(",", ".")
            return pd.to_numeric(text, errors="coerce")  # type: ignore[arg-type]

        return series.apply(_convert)

    def _load_workbook(self, path: Path) -> Dict[str, "pd.Series"]:
        data: Dict[str, "pd.Series"] = {}
        if not path.exists():
            LOGGER.warning("Planilha de preços não encontrada: %s", path)
            return data

        try:
            xls = pd.ExcelFile(path)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - I/O variability
            raise RuntimeError(f"Falha ao abrir planilha de preços {path}: {exc}") from exc

        for sheet_name in xls.sheet_names:
            try:
                df = xls.parse(sheet_name)
            except Exception as exc:  # pragma: no cover - parser variability
                LOGGER.warning(
                    "Falha ao ler a aba %s em %s: %s", sheet_name, path, exc
                )
                continue

            df.columns = [str(col).strip() for col in df.columns]

            column_lookup = {
                self._normalize_column_name(original): original for original in df.columns
            }

            date_key = next(
                (
                    column_lookup[key]
                    for key in column_lookup
                    if key in {"date", "data"}
                ),
                None,
            )

            price_key = next(
                (
                    column_lookup[key]
                    for key in column_lookup
                    if key
                    in {
                        "last price",
                        "fechamento",
                        "ultimo",
                        "último",
                        "close",
                        "preco de fechamento",
                        "preço de fechamento",
                    }
                ),
                None,
            )

            if date_key is None or price_key is None:
                LOGGER.debug(
                    "Aba %s ignorada por não conter as colunas esperadas (encontradas: %s)",
                    sheet_name,
                    ", ".join(df.columns),
                )
                continue

            df = df[[date_key, price_key]].dropna()
            if df.empty:
                continue

            df[date_key] = pd.to_datetime(df[date_key], errors="coerce", dayfirst=True)
            df[price_key] = self._normalize_numeric(df[price_key])
            df = df.dropna(subset=[date_key, price_key])
            if df.empty:
                continue

            series = (
                df.set_index(date_key)[price_key].astype(float).sort_index()
            )
            data[self._normalize_sheet_name(sheet_name)] = series
            LOGGER.debug(
                "Aba %s em %s carregada (%d registros, colunas: data=%s preço=%s)",
                sheet_name,
                path.name,
                len(series),
                date_key,
                price_key,
            )

        return data

    def _build_alias_map(self) -> Dict[str, str]:
        alias: Dict[str, str] = {}
        for canonical in self._bdr_data:
            alias[canonical.upper()] = canonical.upper()

        for bdr, adr in PAIRS_BDR_TO_ADR.items():
            alias.setdefault(bdr.upper(), bdr.upper())
            alias[adr.upper()] = bdr.upper()

        for key, value in _MANUAL_NEWS_ALIASES.items():
            alias[key.upper()] = value.upper()

        return alias

    def covered_bdrs(self) -> List[str]:
        """Return canonical BDR tickers that have historical data available."""

        covered: Dict[str, None] = {}

        for ticker, series in self._bdr_data.items():
            if not series.empty:
                covered[ticker.upper()] = None

        for bdr, adr in PAIRS_BDR_TO_ADR.items():
            adr_series = self._adr_data.get(adr.upper())
            if adr_series is not None and not adr_series.empty:
                covered[bdr.upper()] = None

        return sorted(covered.keys())

    def resolve_ticker(self, ticker: str) -> Optional[str]:
        cleaned = ticker.strip().upper()
        if not cleaned:
            return None

        canonical = self._alias_map.get(cleaned)
        if canonical is None:
            return None

        if self.has_coverage(canonical):
            return canonical

        return None

    def has_coverage(self, canonical: str) -> bool:
        canonical = canonical.upper()
        if canonical in self._bdr_data and not self._bdr_data[canonical].empty:
            return True

        adr = PAIRS_BDR_TO_ADR.get(canonical)
        if adr:
            adr = adr.upper()
            if adr in self._adr_data and not self._adr_data[adr].empty:
                return True

        return False

    def coverage_bounds(self, canonical: str) -> Optional[Tuple[date, date]]:
        canonical = canonical.upper()
        min_dates: List[date] = []
        max_dates: List[date] = []

        series = self._bdr_data.get(canonical)
        if series is not None and not series.empty:
            min_dates.append(series.index.min().date())
            max_dates.append(series.index.max().date())

        adr = PAIRS_BDR_TO_ADR.get(canonical)
        if adr:
            adr_series = self._adr_data.get(adr.upper())
            if adr_series is not None and not adr_series.empty:
                min_dates.append(adr_series.index.min().date())
                max_dates.append(adr_series.index.max().date())

        if not min_dates or not max_dates:
            return None

        return min(min_dates), max(max_dates)

    @staticmethod
    def _slice_series(
        series: "pd.Series", start_date: date, end_date: date
    ) -> "pd.Series":
        mask = (series.index.date >= start_date) & (series.index.date <= end_date)
        return series[mask].sort_index()

    def fetch_series(
        self, canonical_bdr: str, start_date: date, end_date: date
    ) -> Tuple["pd.Series", str]:
        attempted: List[str] = []
        attempt_details: List[str] = []
        canonical = canonical_bdr.upper()

        coverage = self.coverage_bounds(canonical)
        if coverage is not None:
            cov_start, cov_end = coverage
            # Ajusta a janela solicitada para permanecer dentro da cobertura disponível
            attempted_symbols: List[str] = []
            if canonical in self._bdr_data:
                attempted_symbols.append(f"BDR:{canonical}")
            adr_label = PAIRS_BDR_TO_ADR.get(canonical)
            if adr_label:
                attempted_symbols.append(f"ADR:{adr_label}")

            if end_date < cov_start or start_date > cov_end:
                detail_desc = (
                    "Intervalo solicitado %s a %s fora da cobertura disponível %s a %s"
                    % (start_date, end_date, cov_start, cov_end)
                )
                attempted_desc = ", ".join(attempted_symbols) or "nenhuma tentativa"
                raise PriceFetchError(
                    canonical_bdr,
                    attempted_symbols,
                    "Sem dados de preço disponíveis nas planilhas para %s (tentativas: %s)"
                    % (canonical_bdr, attempted_desc),
                    details=detail_desc,
                )

            start_date = max(start_date, cov_start)
            end_date = min(end_date, cov_end)

        if canonical in self._bdr_data:
            attempted.append(f"BDR:{canonical}")
            base_series = self._bdr_data[canonical]
            if base_series.empty:
                attempt_details.append(f"BDR:{canonical} (aba vazia)")
            else:
                series = self._slice_series(base_series, start_date, end_date)
                if not series.empty:
                    return series, f"BDR:{canonical}"
                min_date = base_series.index.min()
                max_date = base_series.index.max()
                attempt_details.append(
                    "BDR:%s (intervalo solicitado %s a %s fora da cobertura disponível %s a %s)"
                    % (
                        canonical,
                        start_date,
                        end_date,
                        min_date.date() if isinstance(min_date, pd.Timestamp) else min_date,
                        max_date.date() if isinstance(max_date, pd.Timestamp) else max_date,
                    )
                )
        else:
            attempt_details.append(f"BDR:{canonical} (aba não encontrada)")

        adr = PAIRS_BDR_TO_ADR.get(canonical)
        if adr:
            adr = adr.upper()
            attempted.append(f"ADR:{adr}")
            if adr in self._adr_data:
                base_series = self._adr_data[adr]
                if base_series.empty:
                    attempt_details.append(f"ADR:{adr} (aba vazia)")
                else:
                    series = self._slice_series(
                        base_series, start_date, end_date
                    )
                    if not series.empty:
                        return series, f"ADR:{adr}"
                    min_date = base_series.index.min()
                    max_date = base_series.index.max()
                    attempt_details.append(
                        "ADR:%s (intervalo solicitado %s a %s fora da cobertura disponível %s a %s)"
                        % (
                            adr,
                            start_date,
                            end_date,
                            min_date.date() if isinstance(min_date, pd.Timestamp) else min_date,
                            max_date.date() if isinstance(max_date, pd.Timestamp) else max_date,
                        )
                    )
            else:
                attempt_details.append(f"ADR:{adr} (aba não encontrada)")

        else:
            mapped = PAIRS_BDR_TO_ADR.get(canonical)
            if mapped:
                attempt_details.append(
                    f"ADR:{mapped.upper()} (aba não encontrada)"
                )
            else:
                attempt_details.append("ADR:sem mapeamento (par ADR inexistente)")

        attempted_desc = ", ".join(attempted) if attempted else "nenhuma tentativa"
        detail_desc = "; ".join(attempt_details) or None
        raise PriceFetchError(
            canonical_bdr,
            attempted,
            "Sem dados de preço disponíveis nas planilhas para %s (tentativas: %s)"
            % (canonical_bdr, attempted_desc),
            details=detail_desc,
        )


_PRICE_STORE: Optional[LocalPriceStore] = None


def _get_price_store() -> LocalPriceStore:
    global _PRICE_STORE
    if _PRICE_STORE is None:
        bdr_candidates = [
            Path("data/hist_bdr.xlsx"),
            Path("data/Hist_BDRs.xlsx"),
        ]
        adr_candidates = [
            Path("data/hist_origem_bdr.xlsx"),
            Path("data/Hist_Origem_BDRs.xlsx"),
            Path("data/Hist_Origem_ADRs.xlsx"),
        ]

        bdr_path = next((path for path in bdr_candidates if path.exists()), bdr_candidates[0])
        adr_path = next((path for path in adr_candidates if path.exists()), adr_candidates[0])

        LOGGER.debug(
            "Inicializando LocalPriceStore com planilhas BDR=%s e Origem=%s",
            bdr_path,
            adr_path,
        )

        _PRICE_STORE = LocalPriceStore(bdr_path, adr_path)
    return _PRICE_STORE


def _fetch_price_series(
    canonical_bdr: str,
    start: datetime,
    end: datetime,
) -> Tuple[pd.Series, str]:
    price_store = _get_price_store()

    today = datetime.utcnow().date()
    start_date = start.date()
    end_date = end.date()

    if start_date > today:
        raise ValueError(
            "Data do evento no futuro: janela inicia em %s, data atual é %s"
            % (start_date, today)
        )

    effective_end_date = min(end_date, today)

    return price_store.fetch_series(canonical_bdr, start_date, effective_end_date)


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
    mismatch_threshold: float = 0.05,
) -> Tuple[
    pd.DataFrame,
    Dict[int, HorizonResult],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if pd is None:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'pandas' é necessário para montar as tabelas de validação. "
            "Instale-o com 'pip install pandas' e rode novamente."
        )
    """Match structured predictions against price direction.

    Returns a tuple with:
        - detailed record dataframe por horizonte
        - horizon summary stats
        - aggregate confusion matrix per horizon
        - availability dataframe with price-fetch diagnostics
        - per-day aggregation (janela de 1 dia por horizonte)
        - multi-news subset (dias com mais de uma notícia)
        - rolling window aggregation (últimos 1/3/5 dias)
    """

    detailed_rows: List[Dict[str, object]] = []
    availability_rows: List[Dict[str, object]] = []
    summaries: Dict[int, HorizonResult] = {}
    confusion_maps: Dict[int, Counter] = defaultdict(Counter)
    event_rows: Dict[str, Dict[str, object]] = {}

    try:
        price_store = _get_price_store()
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    grouped_by_ticker: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        raw = record.get("raw", {})
        ticker = str(raw.get("ticker") or "").strip().upper()
        if not ticker:
            LOGGER.debug("Registro sem ticker, ignorando: %s", record)
            continue
        grouped_by_ticker[ticker].append(record)

    jsonl_tickers = sorted(grouped_by_ticker.keys())
    covered_pairs = []
    uncovered_tickers = []
    for ticker in jsonl_tickers:
        resolved = price_store.resolve_ticker(ticker)
        if resolved is not None:
            covered_pairs.append(f"{ticker}->{resolved}")
        else:
            uncovered_tickers.append(ticker)

    if covered_pairs:
        LOGGER.info(
            "Empresas no JSONL com cobertura nas planilhas: %s",
            ", ".join(covered_pairs),
        )
    else:
        LOGGER.info("Nenhuma empresa do JSONL possui cobertura nas planilhas.")

    if uncovered_tickers:
        LOGGER.info(
            "Empresas no JSONL fora das planilhas: %s",
            ", ".join(uncovered_tickers),
        )
    else:
        LOGGER.info("Todas as empresas do JSONL possuem cobertura nas planilhas.")

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
                availability_rows.append(
                    {
                        "id": raw.get("id"),
                        "ticker": ticker,
                        "country": raw.get("country"),
                        "event_datetime": event_dt,
                        "status": "future_event",
                        "attempted": False,
                        "price_found": False,
                        "message": "Evento em data futura ignorado na validação.",
                    }
                )
                continue

            canonical_ticker = price_store.resolve_ticker(ticker)
            if canonical_ticker is None:
                availability_rows.append(
                    {
                        "id": raw.get("id"),
                        "ticker": ticker,
                        "resolved_ticker": None,
                        "country": raw.get("country"),
                        "event_datetime": event_dt,
                        "status": "ticker_not_covered",
                        "attempted": False,
                        "price_found": False,
                        "message": "Ticker não mapeado para as planilhas de BDR/ADR.",
                    }
                )
                continue

            coverage = price_store.coverage_bounds(canonical_ticker)
            if coverage is not None:
                cov_start, cov_end = coverage
                if event_dt.date() < cov_start or event_dt.date() > cov_end:
                    LOGGER.info(
                        "Evento %s (%s) fora da cobertura disponível para %s: %s a %s",
                        raw.get("id"),
                        ticker,
                        canonical_ticker,
                        cov_start,
                        cov_end,
                    )
                    availability_rows.append(
                        {
                            "id": raw.get("id"),
                            "ticker": ticker,
                            "resolved_ticker": canonical_ticker,
                            "country": raw.get("country"),
                            "event_datetime": event_dt,
                            "status": "event_outside_coverage",
                            "attempted": False,
                            "price_found": False,
                            "message": (
                                "Evento fora da cobertura das planilhas (%s a %s)."
                                % (cov_start, cov_end)
                            ),
                        }
                    )
                    continue
            window_start, window_end = _trading_window(event_dt, max(horizons))

            try:
                price_series, used_symbol = _fetch_price_series(
                    canonical_ticker,
                    window_start,
                    window_end,
                )
            except PriceFetchError as exc:  # pragma: no cover - network variability
                LOGGER.warning(
                    "Falha ao obter preços para %s (%s): %s",
                    ticker,
                    raw.get("id"),
                    exc,
                )
                availability_rows.append(
                    {
                        "id": raw.get("id"),
                        "ticker": ticker,
                        "resolved_ticker": canonical_ticker,
                        "country": raw.get("country"),
                        "event_datetime": event_dt,
                        "status": "price_unavailable",
                        "attempted": True,
                        "price_found": False,
                        "message": str(exc),
                        "attempted_symbols": ", ".join(exc.attempted_symbols),
                    }
                )
                continue
            except Exception as exc:  # pragma: no cover - unexpected failures
                LOGGER.warning("Falha ao obter preços para %s (%s): %s", ticker, raw.get("id"), exc)
                availability_rows.append(
                    {
                        "id": raw.get("id"),
                        "ticker": ticker,
                        "resolved_ticker": canonical_ticker,
                        "country": raw.get("country"),
                        "event_datetime": event_dt,
                        "status": "price_error",
                        "attempted": True,
                        "price_found": False,
                        "message": str(exc),
                    }
                )
                continue

            base = _price_on_or_after(price_series, event_dt)
            if base is None:
                LOGGER.debug("Sem preço base para %s em %s", ticker, event_dt)
                availability_rows.append(
                    {
                        "id": raw.get("id"),
                        "ticker": ticker,
                        "resolved_ticker": canonical_ticker,
                        "country": raw.get("country"),
                        "event_datetime": event_dt,
                        "status": "missing_base_price",
                        "attempted": True,
                        "price_found": False,
                        "message": "Sem preço base na data do evento.",
                        "used_symbol": used_symbol,
                    }
                )
                continue
            base_date, base_price = base

            availability_rows.append(
                {
                    "id": raw.get("id"),
                    "ticker": ticker,
                    "resolved_ticker": canonical_ticker,
                    "country": raw.get("country"),
                    "event_datetime": event_dt,
                    "status": "price_found",
                    "attempted": True,
                    "price_found": True,
                    "message": "",
                    "used_symbol": used_symbol,
                }
            )

            event_id = str(raw.get("id"))
            if event_id and event_id not in event_rows:
                event_rows[event_id] = {
                    "id": event_id,
                    "ticker": ticker,
                    "resolved_ticker": canonical_ticker,
                    "event_datetime": event_dt,
                    "expected_move": expected_move,
                }

            for horizon in horizons:
                target = _price_on_or_after(price_series, event_dt + timedelta(days=horizon))
                if target is None:
                    LOGGER.debug("Sem preço para %s +%sd", ticker, horizon)
                    continue
                target_date, target_price = target
                ret = (target_price - base_price) / base_price
                realized_dir = _direction_from_return(ret, neutral_threshold)
                abs_return = abs(ret)
                sign_mismatch = (expected_move > 0 and ret < 0) or (
                    expected_move < 0 and ret > 0
                )
                within_tolerance = abs_return <= mismatch_threshold

                if realized_dir == expected_move:
                    is_correct = True
                elif sign_mismatch:
                    is_correct = False
                else:
                    is_correct = within_tolerance

                is_neutral_hit = expected_move == 0 and realized_dir == 0
                confusion_maps[horizon][(expected_move, realized_dir)] += 1

                detailed_rows.append(
                    {
                        "id": raw.get("id"),
                        "ticker": ticker,
                        "resolved_ticker": canonical_ticker,
                        "headline": raw.get("headline"),
                        "event_datetime": event_dt,
                        "base_date": base_date,
                        "target_date": target_date,
                        "horizon_days": horizon,
                        "sentiment": sentiment,
                        "expected_move": expected_move,
                        "return": ret,
                        "abs_return": abs_return,
                        "mismatch_threshold": mismatch_threshold,
                        "within_tolerance": within_tolerance,
                        "sign_mismatch": sign_mismatch,
                        "realized_direction": realized_dir,
                        "correct": is_correct,
                        "neutral_hit": is_neutral_hit,
                        "price_source": used_symbol,
                    }
                )

    detailed_df = pd.DataFrame(detailed_rows)
    availability_df = pd.DataFrame(availability_rows)
    if detailed_df.empty:
        LOGGER.warning("Nenhum registro com dados de preço foi avaliado.")
        return (
            detailed_df,
            summaries,
            pd.DataFrame(),
            availability_df,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

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

    per_day_df = _build_per_day_summary(detailed_df)
    multi_news_df = per_day_df[per_day_df["news_count"] > 1].copy()
    rolling_windows_df = _build_rolling_window_summary(
        detailed_df, per_day_df, event_rows.values()
    )

    return (
        detailed_df,
        summaries,
        confusion_df,
        availability_df,
        per_day_df,
        multi_news_df,
        rolling_windows_df,
    )


def _build_per_day_summary(detailed_df: pd.DataFrame) -> pd.DataFrame:
    if detailed_df.empty:
        return pd.DataFrame()

    df = detailed_df.copy()
    df["event_date"] = _ensure_naive_datetime_series(df["event_datetime"]).dt.normalize()

    per_day_rows: List[Dict[str, object]] = []
    grouped = df.groupby(["resolved_ticker", "event_date", "horizon_days"], sort=True)

    for (resolved_ticker, event_date, horizon), group in grouped:
        unique_events = group.drop_duplicates(subset=["id"])
        tickers_on_day = sorted(set(unique_events["ticker"]))

        per_day_rows.append(
            {
                "resolved_ticker": resolved_ticker,
                "event_date": _ensure_naive_timestamp(event_date),
                "horizon_days": int(horizon),
                "news_count": int(unique_events["id"].nunique()),
                "positive_news": int((unique_events["expected_move"] > 0).sum()),
                "negative_news": int((unique_events["expected_move"] < 0).sum()),
                "neutral_news": int((unique_events["expected_move"] == 0).sum()),
                "net_expected_move": float(unique_events["expected_move"].sum()),
                "correct_count": int(group["correct"].sum()),
                "accuracy_rate": float(group["correct"].mean()),
                "any_correct": bool(group["correct"].any()),
                "realized_direction": int(group["realized_direction"].iloc[0]),
                "return": float(group["return"].iloc[0]),
                "price_source": group["price_source"].iloc[0],
                "base_date": group["base_date"].iloc[0],
                "target_date": group["target_date"].iloc[0],
                "tickers_on_day": ",".join(tickers_on_day),
            }
        )

    return pd.DataFrame(per_day_rows)


def _build_rolling_window_summary(
    detailed_df: pd.DataFrame,
    per_day_df: pd.DataFrame,
    events: Iterable[Dict[str, object]],
) -> pd.DataFrame:
    if detailed_df.empty or per_day_df.empty:
        return pd.DataFrame()

    events_list = list(events)
    if not events_list:
        return pd.DataFrame()

    events_df = pd.DataFrame(events_list)
    events_df["event_datetime"] = _ensure_naive_datetime_series(
        events_df["event_datetime"]
    )
    events_df["event_date"] = events_df["event_datetime"].dt.normalize()

    per_day_df = per_day_df.copy()
    per_day_df["event_date"] = _ensure_naive_datetime_series(
        per_day_df["event_date"]
    ).dt.normalize()

    events_by_ticker: Dict[str, pd.DataFrame] = {}
    for ticker, ticker_df in events_df.groupby("resolved_ticker"):
        events_by_ticker[str(ticker)] = ticker_df.sort_values("event_datetime")

    per_day_index = per_day_df.set_index(
        ["resolved_ticker", "event_date", "horizon_days"]
    )

    window_rows: List[Dict[str, object]] = []

    for (resolved_ticker, event_date, horizon), row in per_day_index.iterrows():
        ticker_events = events_by_ticker.get(str(resolved_ticker))
        if ticker_events is None or ticker_events.empty:
            continue

        event_date_ts = _ensure_naive_timestamp(event_date)

        for window_size in ROLLING_WINDOW_SIZES:
            window_start = event_date_ts - pd.Timedelta(days=window_size - 1)
            mask = (ticker_events["event_date"] >= window_start) & (
                ticker_events["event_date"] <= event_date_ts
            )
            window_events = ticker_events.loc[mask]
            if window_events.empty:
                continue

            net_expected = float(window_events["expected_move"].sum())
            expected_direction = 1 if net_expected > 0 else -1 if net_expected < 0 else 0

            realized_direction = int(row["realized_direction"])
            correct = realized_direction == expected_direction
            neutral_hit = expected_direction == 0 and realized_direction == 0

            window_rows.append(
                {
                    "resolved_ticker": resolved_ticker,
                    "event_date": event_date_ts,
                    "window_start_date": window_start,
                    "window_size_days": int(window_size),
                    "horizon_days": int(horizon),
                    "news_count_window": int(len(window_events)),
                    "news_on_event_date": int(row["news_count"]),
                    "positive_news_window": int(
                        (window_events["expected_move"] > 0).sum()
                    ),
                    "negative_news_window": int(
                        (window_events["expected_move"] < 0).sum()
                    ),
                    "neutral_news_window": int(
                        (window_events["expected_move"] == 0).sum()
                    ),
                    "net_expected_move_window": net_expected,
                    "expected_direction_window": int(expected_direction),
                    "realized_direction": realized_direction,
                    "return": float(row["return"]),
                    "correct": bool(correct),
                    "neutral_hit": bool(neutral_hit),
                    "price_source": row.get("price_source"),
                    "tickers_on_day": row.get("tickers_on_day"),
                }
            )

    return pd.DataFrame(window_rows)


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


def generate_multi_news_plot(multi_df: pd.DataFrame, output_path: Path) -> None:
    if multi_df.empty:
        LOGGER.warning(
            "Sem dados de múltiplas notícias no mesmo dia para gerar gráfico."
        )
        return

    summary = (
        multi_df.groupby("horizon_days")["accuracy_rate"].mean().reset_index()
    )
    summary.rename(columns={"accuracy_rate": "accuracy"}, inplace=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'matplotlib' é necessário para gerar gráficos. "
            "Instale-o com 'pip install matplotlib' e rode novamente."
        ) from exc

    plt.figure(figsize=(8, 5))
    plt.bar(summary["horizon_days"], summary["accuracy"], color="#9467bd")
    plt.ylim(0, 1)
    plt.xlabel("Horizonte (dias)")
    plt.ylabel("Acurácia média")
    plt.title("Acurácia para dias com múltiplas notícias")
    for x, acc in zip(summary["horizon_days"], summary["accuracy"]):
        label = "N/A" if math.isnan(acc) else f"{acc:.1%}"
        plt.text(x, min(acc if not math.isnan(acc) else 0.0, 0.95), label, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_rolling_window_plot(rolling_df: pd.DataFrame, output_path: Path) -> None:
    if rolling_df.empty:
        LOGGER.warning(
            "Sem dados agregados por janela móvel para gerar gráfico."
        )
        return

    accuracy = (
        rolling_df.groupby(["window_size_days", "horizon_days"])["correct"]
        .mean()
        .reset_index()
    )
    pivot = accuracy.pivot(
        index="window_size_days", columns="horizon_days", values="correct"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'matplotlib' é necessário para gerar gráficos. "
            "Instale-o com 'pip install matplotlib' e rode novamente."
        ) from exc

    try:
        import seaborn as sns
    except ImportError:
        LOGGER.warning(
            "Seaborn não instalado; gráfico de janela móvel será renderizado como mapa simples."
        )
        plt.figure(figsize=(8, 5))
        plt.imshow(pivot, cmap="Purples", aspect="auto")
        plt.colorbar(label="Acurácia média")
        plt.xticks(
            range(len(pivot.columns)),
            [str(col) for col in pivot.columns],
        )
        plt.yticks(
            range(len(pivot.index)),
            [str(idx) for idx in pivot.index],
        )
        plt.xlabel("Horizonte (dias)")
        plt.ylabel("Janela de notícias (dias)")
        plt.title("Acurácia por janela de notícias e horizonte")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Purples", vmin=0, vmax=1)
    plt.xlabel("Horizonte (dias)")
    plt.ylabel("Janela de notícias (dias)")
    plt.title("Acurácia por janela de notícias e horizonte")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_price_availability_plot(
    availability_df: pd.DataFrame, output_path: Path
) -> None:
    if availability_df.empty:
        LOGGER.warning("Sem dados de disponibilidade de preço para gerar gráfico.")
        return

    attempted_df = availability_df[availability_df["attempted"]]
    if attempted_df.empty:
        LOGGER.warning(
            "Nenhuma tentativa de busca de preço registrada; gráfico de disponibilidade não será gerado."
        )
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "O pacote 'matplotlib' é necessário para gerar gráficos. "
            "Instale-o com 'pip install matplotlib' e rode novamente."
        ) from exc

    summary = (
        attempted_df["price_found"].value_counts().rename(index={True: "Preço encontrado", False: "Preço indisponível"})
    )

    labels = summary.index.tolist()
    values = summary.values.tolist()

    plt.figure(figsize=(6, 4))
    colors = ["#2ca02c" if "encontrado" in label.lower() else "#d62728" for label in labels]
    plt.bar(labels, values, color=colors)
    plt.title("Disponibilidade de preços nas validações")
    plt.ylabel("Quantidade de eventos")
    for idx, value in enumerate(values):
        plt.text(idx, value + 0.05 * max(values), str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

