"""Real-time dissemination features inspired by the FinGPT paper."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


SentimentFn = Callable[[str], float]


def _default_sentiment(text: str) -> float:
    positive_keywords = ("record", "supera", "expands", "beats")
    negative_keywords = ("queda", "miss", "atraso", "demissão", "recall")
    score = 0.0
    lowered = text.lower()
    for word in positive_keywords:
        if word in lowered:
            score += 1.0
    for word in negative_keywords:
        if word in lowered:
            score -= 1.0
    return score


@dataclass
class ClusterFeatureEngineer:
    window_minutes: int = 15
    max_clusters: int = 8
    min_articles_per_window: int = 2
    sentiment_fn: SentimentFn = _default_sentiment

    def transform(self, news: pd.DataFrame, text_column: str = "text", timestamp_column: str = "timestamp") -> pd.DataFrame:
        if text_column not in news or timestamp_column not in news:
            raise ValueError(f"DataFrame must contain '{text_column}' and '{timestamp_column}' columns.")

        df = news[[timestamp_column, text_column]].copy()
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df.sort_values(timestamp_column, inplace=True)
        df.dropna(subset=[text_column], inplace=True)

        df["sentiment_score"] = df[text_column].apply(self.sentiment_fn)
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)

        features: List[Dict[str, float]] = []
        previous_counts: Dict[int, int] = defaultdict(int)

        if df.empty:
            return pd.DataFrame(columns=[
                "window_start",
                "window_end",
                "numero_clusters_ativos",
                "tamanho_maior_cluster",
                "velocidade_clusters",
                "sentimento_ponderado_cluster",
            ])

        window_delta = timedelta(minutes=self.window_minutes)
        window_end = df[timestamp_column].min() + window_delta
        while window_end <= df[timestamp_column].max() + window_delta:
            window_start = window_end - window_delta
            mask = (df[timestamp_column] > window_start) & (df[timestamp_column] <= window_end)
            window_df = df.loc[mask]

            if len(window_df) < self.min_articles_per_window:
                features.append({
                    "window_start": window_start,
                    "window_end": window_end,
                    "numero_clusters_ativos": 0,
                    "tamanho_maior_cluster": 0,
                    "velocidade_clusters": 0.0,
                    "sentimento_ponderado_cluster": 0.0,
                })
                window_end += window_delta
                continue

            docs = window_df[text_column].tolist()
            tfidf_matrix = vectorizer.fit_transform(docs)
            n_clusters = min(self.max_clusters, len(docs))
            if n_clusters <= 1:
                labels = np.zeros(len(docs), dtype=int)
            else:
                model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(tfidf_matrix)

            counts = Counter(labels)
            numero_clusters = len(counts)
            tamanho_maior = max(counts.values()) if counts else 0

            velocidade_total = 0.0
            for label, count in counts.items():
                previous = previous_counts.get(label, 0)
                velocidade_total += max(count - previous, 0) / self.window_minutes
                previous_counts[label] = count

            cluster_sentiment = 0.0
            labels_array = np.array(labels)
            for label, count in counts.items():
                cluster_mask = labels_array == label
                mean_sentiment = window_df.iloc[cluster_mask]["sentiment_score"].mean()
                cluster_sentiment += mean_sentiment * count
            if sum(counts.values()):
                cluster_sentiment /= sum(counts.values())

            features.append({
                "window_start": window_start,
                "window_end": window_end,
                "numero_clusters_ativos": numero_clusters,
                "tamanho_maior_cluster": tamanho_maior,
                "velocidade_clusters": velocidade_total,
                "sentimento_ponderado_cluster": cluster_sentiment,
            })

            window_end += window_delta

        return pd.DataFrame(features)
