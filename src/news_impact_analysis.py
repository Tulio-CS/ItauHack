"""Tools for analysing the impact of news on stock tickers.

This module implements a multi-stage pipeline inspired by the user's
specification:

1. A relevance filter driven by an LLM classifies each article as
   *Market-Moving*, *Fluff/Marketing* or *Irrelevant*.
2. For market-moving articles, the LLM produces a structured JSON event
   description that captures metrics, expectations and sentiment.
3. A chain-of-thought style prompt extracts an impact score (1-10) and a
   natural-language justification, enabling richer numerical features.
4. News items are grouped in near-real time via clustering so the model can
   reason about information dissemination (FinGPT idea).
5. Price history is downloaded automatically (via yfinance) and used to
   compute forward returns over multiple horizons (1d, 3d, 5d).
6. A gradient boosted tree model (XGBoost) learns the relationship between
   extracted features and realised returns.

The resulting pipeline can be used to train a model or to score incoming
articles for a specific ticker.

The code is deliberately modular so that each step can be swapped or extended
in the future.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

try:  # Optional dependency – only required when using the real API
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback when OpenAI SDK is missing
    OpenAI = None  # type: ignore

try:  # yfinance may not be installed at import time
    import yfinance as yf
except Exception:  # pragma: no cover - allow the module to be imported without it
    yf = None  # type: ignore


LOGGER = logging.getLogger(__name__)


MARKET_MOVING_PROMPT = """
Classifique a notícia abaixo em uma das três categorias: Market-Moving,
Fluff/Marketing ou Irrelevante. Use a seguinte definição:
- Market-Moving: notícias que podem impactar diretamente o valuation da empresa
  (lucros, guidance, lançamento relevante, processos, demissões, fusões etc.).
- Fluff/Marketing: divulgação promocional, entrevistas genéricas, filantropia
  sem impacto financeiro direto.
- Irrelevante: textos que nada têm a ver com a empresa ou o mercado financeiro.

Responda apenas com o rótulo exato (Market-Moving, Fluff/Marketing ou
Irrelevante).

Notícia:
{conteudo}
""".strip()


EVENT_EXTRACTION_PROMPT = """
Você é um analista financeiro. Converta a notícia abaixo para um JSON que siga
exatamente o seguinte formato (sem comentários adicionais):
{
  "evento_tipo": "...",
  "metricas": [
    {"metrica": "...", "valor": <float ou null>, "expectativa": <float ou null>, "resultado": "beat/miss/in-line/na"}
  ],
  "sentimento_geral": "positivo/negativo/misto/neutro"
}

Para valores numéricos, use ponto como separador decimal e somente números sem
símbolos adicionais. Caso alguma informação não exista, use null e "na".

Notícia:
{conteudo}
""".strip()


IMPACT_RATING_PROMPT = """
Analise a notícia abaixo passo a passo:
1. Identifique o evento principal.
2. Avalie o impacto potencial no ticker em uma escala de 1 (nenhum impacto)
   a 10 (impacto máximo).
3. Justifique em duas frases objetivas o motivo da nota.

Responda estritamente no formato JSON:
{
  "evento_principal": "...",
  "nota_de_impacto": <inteiro 1-10>,
  "justificativa": "...",
  "sentimento": "positivo/negativo/misto/neutro"
}

Notícia:
{conteudo}
""".strip()


@dataclass
class NewsRecord:
    """Structured representation of a single news item."""

    ticker: str
    title: str
    content: str
    published_at: datetime
    source: Optional[str] = None


@dataclass
class StructuredEvent:
    """Output of the LLM event extraction step."""

    event_type: str
    metrics: List[Dict[str, Any]]
    overall_sentiment: str


@dataclass
class ImpactAssessment:
    """Impact score and reasoning provided by the LLM."""

    main_event: str
    impact_score: int
    justification: str
    sentiment: str


class LLMAnalyzer:
    """Wrapper responsible for all language model calls.

    The class gracefully falls back to heuristic rules when the OpenAI SDK is
    not available or when no API key is provided, allowing offline usage for
    testing and unit tests.
    """

    def __init__(self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key and OpenAI is not None:
            self._client = OpenAI(api_key=self.api_key)
        else:
            self._client = None
            LOGGER.warning(
                "OpenAI API key não configurada ou SDK indisponível. Usando heurísticas simples."
            )

    def _call_llm(self, prompt: str) -> str:
        if not self._client:
            raise RuntimeError("LLM client not available")

        response = self._client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=400,
        )
        # The Responses API returns a complex object; extract the first text block.
        for item in response.output:  # type: ignore[attr-defined]
            if item["type"] == "message":
                for content in item["content"]:
                    if content["type"] == "text":
                        return content["text"]
        # Fallback: the SDK may expose a convenience property
        if hasattr(response, "output_text"):
            return response.output_text  # type: ignore[attr-defined]
        raise RuntimeError("Não foi possível extrair a resposta do LLM")

    # --- Public API -----------------------------------------------------

    def classify_news(self, text: str) -> str:
        try:
            response_text = self._call_llm(MARKET_MOVING_PROMPT.format(conteudo=text))
            return response_text.strip()
        except Exception:
            # Heuristic fallback: check for financial keywords
            lowered = text.lower()
            triggers = [
                "lucro",
                "prejuízo",
                "guidance",
                "demissão",
                "processo",
                "fusao",
                "aquisição",
                "ipo",
                "resultado",
                "recompra",
            ]
            if any(token in lowered for token in triggers):
                return "Market-Moving"
            if "evento" in lowered or "marketing" in lowered:
                return "Fluff/Marketing"
            return "Irrelevante"

    def extract_structured_event(self, text: str) -> StructuredEvent:
        try:
            response_text = self._call_llm(EVENT_EXTRACTION_PROMPT.format(conteudo=text))
            data = json.loads(response_text)
        except Exception:
            data = {
                "evento_tipo": "unknown",
                "metricas": [],
                "sentimento_geral": "neutro",
            }
        return StructuredEvent(
            event_type=data.get("evento_tipo", "unknown"),
            metrics=data.get("metricas", []),
            overall_sentiment=data.get("sentimento_geral", "neutro"),
        )

    def assess_impact(self, text: str) -> ImpactAssessment:
        try:
            response_text = self._call_llm(IMPACT_RATING_PROMPT.format(conteudo=text))
            data = json.loads(response_text)
        except Exception:
            data = {
                "evento_principal": "unknown",
                "nota_de_impacto": 5,
                "justificativa": "Estimativa heurística por ausência do LLM.",
                "sentimento": "neutro",
            }
        score = int(data.get("nota_de_impacto", 5))
        score = max(1, min(score, 10))
        return ImpactAssessment(
            main_event=data.get("evento_principal", "unknown"),
            impact_score=score,
            justification=data.get("justificativa", ""),
            sentiment=data.get("sentimento", "neutro"),
        )


class NewsLoader:
    """Utility responsible for reading and normalising the news datasets."""

    def __init__(self, data_paths: Sequence[Path]) -> None:
        self.data_paths = list(data_paths)

    @staticmethod
    def _normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        candidates = [
            "published_at",
            "date",
            "published_date",
            "datetime",
        ]
        date_col = next((c for c in candidates if c in df.columns), None)
        if date_col is None:
            raise ValueError("Nenhuma coluna de data encontrada no dataset")

        df = df.copy()
        df["published_at"] = df[date_col].apply(NewsLoader._parse_date)

        ticker_col = next((c for c in ("ticker", "symbol", "ativo") if c in df.columns), None)
        if ticker_col is None:
            raise ValueError("Nenhuma coluna de ticker encontrada no dataset")

        content_col = next((c for c in ("content", "body", "texto") if c in df.columns), None)
        if content_col is None:
            raise ValueError("Nenhuma coluna de conteúdo encontrada")

        title_col = next((c for c in ("title", "headline", "titulo") if c in df.columns), None)
        if title_col is None:
            title_col = content_col

        df = df.rename(
            columns={
                ticker_col: "ticker",
                content_col: "content",
                title_col: "title",
            }
        )
        if "source" not in df.columns:
            df["source"] = None
        return df[["ticker", "title", "content", "published_at", "source"]]

    @staticmethod
    def _parse_date(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            return parser.parse(value)
        raise TypeError(f"Tipo de data não suportado: {type(value)}")

    def load(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for path in self.data_paths:
            if not path.exists():
                LOGGER.warning("Arquivo %s não encontrado; ignorando", path)
                continue
            df = pd.read_parquet(path)
            frames.append(self._normalise_dataframe(df))
        if not frames:
            raise FileNotFoundError("Nenhum dataset válido foi carregado")
        combined = pd.concat(frames, ignore_index=True)
        combined.dropna(subset=["ticker", "content", "published_at"], inplace=True)
        combined.sort_values("published_at", inplace=True)
        return combined


class DisseminationFeatureBuilder:
    """Implements the FinGPT-style clustering features."""

    def __init__(self, n_features: int = 5000, batch_size: int = 64, random_state: int = 42) -> None:
        self.vectorizer = TfidfVectorizer(max_features=n_features)
        self.batch_size = batch_size
        self.random_state = random_state

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.vectorizer.transform(texts)

    def build_features(
        self,
        df: pd.DataFrame,
        window_minutes: int = 30,
        max_clusters: int = 8,
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[
                "numero_clusters_ativos",
                "tamanho_maior_cluster",
                "velocidade_cluster",
                "sentimento_ponderado_cluster",
            ])

        vectors = self.fit_transform(df["content"].tolist())
        km = MiniBatchKMeans(
            n_clusters=min(max_clusters, max(1, vectors.shape[0] // 2)),
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        cluster_labels = km.fit_predict(vectors)

        df = df.copy()
        df["cluster"] = cluster_labels
        df.sort_values("published_at", inplace=True)
        features: List[Dict[str, Any]] = []
        window = timedelta(minutes=window_minutes)

        for i, row in df.iterrows():
            start_time = row["published_at"] - window
            mask = (df["published_at"] >= start_time) & (df["published_at"] <= row["published_at"])
            window_df = df.loc[mask]
            cluster_counts = window_df["cluster"].value_counts()
            numero_clusters_ativos = cluster_counts.shape[0]
            tamanho_maior_cluster = cluster_counts.max()

            # Estimate cluster velocity by comparing to previous window
            prev_mask = (df["published_at"] >= start_time - window) & (df["published_at"] < start_time)
            prev_counts = df.loc[prev_mask, "cluster"].value_counts()
            if not prev_counts.empty:
                growth = (cluster_counts.sum() - prev_counts.sum()) / max(prev_counts.sum(), 1)
            else:
                growth = cluster_counts.sum()

            # Weighted sentiment using the impact assessment's sentiment when available
            sentiments = window_df.get("impact_sentiment", pd.Series(["neutro"] * len(window_df)))
            weights = window_df.groupby("cluster").cluster.transform("count")
            sentiment_map = {
                "positivo": 1.0,
                "negativo": -1.0,
                "misto": 0.0,
                "neutro": 0.0,
            }
            sent_values = sentiments.map(sentiment_map).fillna(0.0)
            if weights is not None and weights.sum() > 0:
                weighted_sentiment = float(np.average(sent_values, weights=weights))
            else:
                weighted_sentiment = 0.0

            features.append(
                {
                    "index": i,
                    "numero_clusters_ativos": float(numero_clusters_ativos),
                    "tamanho_maior_cluster": float(tamanho_maior_cluster),
                    "velocidade_cluster": float(growth),
                    "sentimento_ponderado_cluster": weighted_sentiment,
                }
            )

        feature_df = pd.DataFrame(features).set_index("index")
        return feature_df


class FeatureAssembler:
    """Builds the machine-learning ready feature matrix."""

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        categorical_cols = ["evento_tipo", "sentimento_geral", "impact_sentiment"]
        numeric_cols = [
            "impact_score",
            "numero_clusters_ativos",
            "tamanho_maior_cluster",
            "velocidade_cluster",
            "sentimento_ponderado_cluster",
        ]
        metric_features = self._expand_metric_features(df.get("metricas", pd.Series(dtype=object)))
        numeric = df[numeric_cols].fillna(0.0)
        categorical = df[categorical_cols].fillna("desconhecido")
        encoded = self.encoder.fit_transform(categorical).toarray()
        self._fitted = True
        return np.hstack([numeric.values, metric_features, encoded])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder não foi treinado")
        categorical_cols = ["evento_tipo", "sentimento_geral", "impact_sentiment"]
        numeric_cols = [
            "impact_score",
            "numero_clusters_ativos",
            "tamanho_maior_cluster",
            "velocidade_cluster",
            "sentimento_ponderado_cluster",
        ]
        metric_features = self._expand_metric_features(df.get("metricas", pd.Series(dtype=object)))
        numeric = df[numeric_cols].fillna(0.0)
        categorical = df[categorical_cols].fillna("desconhecido")
        encoded = self.encoder.transform(categorical).toarray()
        return np.hstack([numeric.values, metric_features, encoded])

    @staticmethod
    def _expand_metric_features(metric_series: pd.Series) -> np.ndarray:
        # Flatten metrics into aggregated statistics (count of beats/misses, avg surprise etc.)
        beat_counts = []
        miss_counts = []
        inline_counts = []
        surprises = []
        for metrics in metric_series.fillna([]):
            beats = sum(1 for m in metrics if m.get("resultado") == "beat")
            misses = sum(1 for m in metrics if m.get("resultado") == "miss")
            inline = sum(1 for m in metrics if m.get("resultado") == "in-line")
            surprises.append(FeatureAssembler._average_surprise(metrics))
            beat_counts.append(beats)
            miss_counts.append(misses)
            inline_counts.append(inline)
        return np.vstack([beat_counts, miss_counts, inline_counts, surprises]).T

    @staticmethod
    def _average_surprise(metrics: Sequence[Dict[str, Any]]) -> float:
        values: List[float] = []
        for metric in metrics:
            try:
                actual = float(metric.get("valor")) if metric.get("valor") is not None else None
                expected = float(metric.get("expectativa")) if metric.get("expectativa") is not None else None
            except (TypeError, ValueError):
                actual = expected = None
            if actual is not None and expected not in (None, 0):
                values.append((actual - expected) / abs(expected))
        if not values:
            return 0.0
        return float(np.mean(values))


class PriceFetcher:
    """Downloads price data from yfinance."""

    def __init__(self, auto_adjust: bool = True) -> None:
        if yf is None:
            raise ImportError("yfinance não está instalado. Instale antes de usar o PriceFetcher.")
        self.auto_adjust = auto_adjust

    def get_history(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=self.auto_adjust)
        if data.empty:
            raise ValueError(f"Sem dados de preço para {ticker}")
        data = data[["Close"]].rename(columns={"Close": "close"})
        data.index = pd.to_datetime(data.index)
        return data


class XGBoostImpactModel:
    """Wraps an XGBoost regressor to predict future returns."""

    def __init__(self) -> None:
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            objective="reg:squarederror",
        )
        self.features = FeatureAssembler()
        self._trained = False

    def fit(self, df: pd.DataFrame, target_column: str) -> None:
        X = self.features.fit_transform(df)
        y = df[target_column].values
        self.model.fit(X, y)
        self._trained = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Modelo ainda não treinado")
        X = self.features.transform(df)
        return self.model.predict(X)


class NewsImpactAnalyzer:
    """High-level orchestration class."""

    def __init__(
        self,
        data_paths: Sequence[Path],
        llm_model: str = "gpt-4.1-mini",
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.loader = NewsLoader(data_paths)
        self.llm = LLMAnalyzer(model=llm_model, api_key=openai_api_key)
        self.cluster_builder = DisseminationFeatureBuilder()
        self.price_fetcher = PriceFetcher()
        self.model = XGBoostImpactModel()

    def prepare_dataset(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        news_df = self.loader.load()
        news_df = news_df[news_df["ticker"].str.upper() == ticker.upper()].copy()
        if start:
            news_df = news_df[news_df["published_at"] >= start]
        if end:
            news_df = news_df[news_df["published_at"] <= end]
        if news_df.empty:
            raise ValueError(f"Nenhuma notícia encontrada para {ticker}")

        records: List[Dict[str, Any]] = []
        for _, row in news_df.iterrows():
            text = f"{row['title']}\n\n{row['content']}"
            classification = self.llm.classify_news(text)
            if classification != "Market-Moving":
                continue
            structured = self.llm.extract_structured_event(text)
            impact = self.llm.assess_impact(text)
            records.append(
                {
                    "ticker": ticker,
                    "title": row["title"],
                    "content": row["content"],
                    "published_at": row["published_at"],
                    "evento_tipo": structured.event_type,
                    "metricas": structured.metrics,
                    "sentimento_geral": structured.overall_sentiment,
                    "impact_score": impact.impact_score,
                    "impact_sentiment": impact.sentiment,
                    "impact_justificativa": impact.justification,
                }
            )

        dataset = pd.DataFrame(records)
        if dataset.empty:
            raise ValueError("Nenhuma notícia market-moving encontrada")

        dissemination_features = self.cluster_builder.build_features(dataset)
        dataset = dataset.join(dissemination_features, how="left")

        price_start = dataset["published_at"].min() - timedelta(days=2)
        price_end = dataset["published_at"].max() + timedelta(days=10)
        price_history = self.price_fetcher.get_history(ticker, price_start, price_end)

        returns = []
        for _, row in dataset.iterrows():
            event_time = row["published_at"].floor("D")
            base_price = self._get_closest_price(price_history, event_time)
            future_returns = {
                "ret_1d": self._compute_return(price_history, event_time, 1, base_price),
                "ret_3d": self._compute_return(price_history, event_time, 3, base_price),
                "ret_5d": self._compute_return(price_history, event_time, 5, base_price),
            }
            returns.append(future_returns)
        returns_df = pd.DataFrame(returns, index=dataset.index)
        dataset = pd.concat([dataset, returns_df], axis=1)
        return dataset

    def train_model(self, dataset: pd.DataFrame, horizon: str = "ret_3d") -> None:
        if horizon not in {"ret_1d", "ret_3d", "ret_5d"}:
            raise ValueError("Horizonte inválido")
        self.model.fit(dataset, target_column=horizon)

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        predictions = {
            "pred_ret_1d": self.model.predict(dataset.assign(ret_1d=0, ret_3d=0, ret_5d=0)),
        }
        result = dataset.copy()
        for name, values in predictions.items():
            result[name] = values
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _get_closest_price(price_history: pd.DataFrame, date: datetime) -> float:
        # Attempt to get the closing price on the date; fallback to next available
        if date in price_history.index:
            return float(price_history.loc[date, "close"])
        future_prices = price_history[price_history.index >= date]
        if future_prices.empty:
            raise ValueError(f"Sem preço disponível após {date}")
        return float(future_prices.iloc[0]["close"])

    @staticmethod
    def _compute_return(
        price_history: pd.DataFrame,
        event_time: datetime,
        days_ahead: int,
        base_price: float,
    ) -> float:
        target_date = event_time + timedelta(days=days_ahead)
        if target_date in price_history.index:
            future_price = float(price_history.loc[target_date, "close"])
        else:
            future_prices = price_history[price_history.index >= target_date]
            if future_prices.empty:
                return np.nan
            future_price = float(future_prices.iloc[0]["close"])
        if base_price in (0, np.nan):
            return np.nan
        return (future_price - base_price) / base_price


def build_default_analyzer(data_dir: Path) -> NewsImpactAnalyzer:
    """Convenience factory that points to the default parquet files."""

    paths = [
        data_dir / "investing_news.parquet",
        data_dir / "investing_news_nacionais.parquet",
        data_dir / "investing_news_nacionais_que_faltaram.parquet",
    ]
    return NewsImpactAnalyzer(paths)


__all__ = [
    "NewsRecord",
    "StructuredEvent",
    "ImpactAssessment",
    "LLMAnalyzer",
    "NewsLoader",
    "DisseminationFeatureBuilder",
    "FeatureAssembler",
    "PriceFetcher",
    "XGBoostImpactModel",
    "NewsImpactAnalyzer",
    "build_default_analyzer",
]
