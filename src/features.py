"""Helpers to convert structured events into ML ready features."""
from __future__ import annotations

from collections import Counter
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

SENTIMENT_TO_NUMERIC = {"negativo": -1, "misto": 0, "neutro": 0, "positivo": 1}

logger = logging.getLogger(__name__)


def extract_metric_features(metricas: List[Dict[str, str]]) -> Dict[str, float]:
    results = Counter()
    for metrica in metricas:
        resultado = (metrica.get("resultado") or "desconhecido").lower()
        results[f"metric_{resultado}"] += 1
    return {
        "metric_count": float(sum(results.values())),
        "metric_beat": float(results.get("metric_beat", 0)),
        "metric_miss": float(results.get("metric_miss", 0)),
        "metric_inline": float(results.get("metric_inline", 0)),
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features: List[Dict[str, float]] = []
    logger.info("Construindo matriz de features para %s notícias", len(df))
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        numeric_sentiment = SENTIMENT_TO_NUMERIC.get(row.sentimento_geral.lower(), 0)
        metric_features = extract_metric_features(getattr(row, "metricas", []) or [])
        record = {
            "impacto_nota": float(getattr(row, "impacto_nota", 0.0) or 0.0),
            "sentimento_num": numeric_sentiment,
        }
        record.update(metric_features)
        # one-hot for event type
        evento_tipo = getattr(row, "evento_tipo", "outros") or "outros"
        record[f"evento_{evento_tipo.lower().strip().replace(' ', '_')}"] = 1.0
        features.append(record)
        if idx % 100 == 0:
            logger.debug("%s registros de features processados", idx)
    feature_df = pd.DataFrame(features).fillna(0.0)
    logger.info("Matriz de features final possui shape %s", feature_df.shape)
    return feature_df


def align_features(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> pd.DataFrame:
    for column in train_df.columns:
        if column not in predict_df.columns:
            predict_df[column] = 0.0
    extra_columns = [c for c in predict_df.columns if c not in train_df.columns]
    if extra_columns:
        predict_df = predict_df.drop(columns=extra_columns)
    return predict_df[train_df.columns]


def compute_price_impact(prices: pd.DataFrame) -> float:
    close = prices["Close"].astype(float)
    if len(close) < 2:
        logger.debug("Preço insuficiente para calcular impacto. Registros=%s", len(close))
        return 0.0
    start = close.iloc[0]
    end = close.iloc[-1]
    impact = float((end - start) / start)
    logger.debug("Impacto de preço calculado: início=%s fim=%s impacto=%.5f", start, end, impact)
    return impact


def compute_return_label(returns: float, threshold: float = 0.0) -> int:
    if returns > threshold:
        return 1
    if returns < -threshold:
        return -1
    return 0
