"""Composable helpers to load, filter and enrich financial news datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .cluster_features import ClusterFeatureEngineer
from .event_extractor import StructuredEventExtractor
from .llm_interface import LLMClient, RuleBasedLLMClient
from .relevance_filter import NewsCategory, NewsRelevanceFilter


def load_news(
    path: Path,
    text_column: str = "headline",
    timestamp_column: str = "created_at",
) -> pd.DataFrame:
    """Load a CSV/Parquet file and validate the expected columns."""

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Formato de arquivo não suportado. Use CSV ou Parquet.")

    missing: List[str] = []
    if text_column not in df:
        missing.append(text_column)
    if timestamp_column not in df:
        missing.append(timestamp_column)
    if missing:
        raise ValueError(
            "Colunas ausentes no dataset: " + ", ".join(sorted(set(missing)))
        )

    return df


def relevance_filter(
    df: pd.DataFrame,
    text_column: str,
    llm_client: Optional[LLMClient] = None,
) -> pd.DataFrame:
    """Classify each news item and keep only market-moving entries."""

    client = llm_client or RuleBasedLLMClient()
    filterer = NewsRelevanceFilter(client)
    categories: List[NewsCategory] = []
    for text in df[text_column]:
        categories.append(filterer.classify(str(text)))

    filtered = df.copy()
    filtered["news_category"] = categories
    return filtered[filtered["news_category"] == NewsCategory.MARKET_MOVING]


def extract_events(
    df: pd.DataFrame,
    text_column: str,
    llm_client: Optional[LLMClient] = None,
) -> pd.DataFrame:
    """Generate structured events and flatten metric-level features."""

    extractor = StructuredEventExtractor(llm_client)

    events: List[Dict[str, object]] = []
    event_types: List[str] = []
    sentiments: List[str] = []
    impact_scores: List[int | None] = []
    impact_rationales: List[str | None] = []
    metric_features: List[Dict[str, object]] = []

    for text in df[text_column]:
        event = extractor.extract(str(text))
        events.append(event.to_dict())
        event_types.append(event.event_type)
        sentiments.append(event.overall_sentiment)
        impact_scores.append(event.impact_rating)
        impact_rationales.append(event.impact_rationale)

        metrics_dict: Dict[str, object] = {}
        for metric in event.metrics:
            prefix = metric.metric.lower().replace(" ", "_")
            if metric.value is not None:
                metrics_dict[f"{prefix}_valor"] = metric.value
            if metric.expectation is not None:
                metrics_dict[f"{prefix}_expectativa"] = metric.expectation
            if metric.outcome:
                metrics_dict[f"{prefix}_resultado"] = metric.outcome
        metric_features.append(metrics_dict)

    enriched = df.copy()
    enriched["structured_event"] = events
    enriched["evento_tipo"] = event_types
    enriched["sentimento_geral"] = sentiments
    enriched["impacto_potencial"] = impact_scores
    enriched["justificativa_impacto"] = impact_rationales

    metrics_df = pd.DataFrame(metric_features)
    if not metrics_df.empty:
        enriched = pd.concat([enriched.reset_index(drop=True), metrics_df], axis=1)
    else:
        enriched = enriched.reset_index(drop=True)

    return enriched


def compute_cluster_features(
    df: pd.DataFrame,
    text_column: str,
    timestamp_column: str,
    engineer: Optional[ClusterFeatureEngineer] = None,
) -> pd.DataFrame:
    """Derive dissemination metrics using the FinGPT-inspired engineer."""

    engineer = engineer or ClusterFeatureEngineer()
    return engineer.transform(df, text_column=text_column, timestamp_column=timestamp_column)


def run_pipeline(
    input_path: Path,
    text_column: str = "headline",
    timestamp_column: str = "created_at",
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, pd.DataFrame]:
    """Execute the full filtering, extraction and cluster feature pipeline."""

    news = load_news(input_path, text_column=text_column, timestamp_column=timestamp_column)
    filtered = relevance_filter(news, text_column, llm_client=llm_client)
    if filtered.empty:
        return {
            "filtered_news": filtered,
            "cluster_features": pd.DataFrame(
                columns=[
                    "window_start",
                    "window_end",
                    "numero_clusters_ativos",
                    "tamanho_maior_cluster",
                    "velocidade_clusters",
                    "sentimento_ponderado_cluster",
                ]
            ),
        }

    enriched = extract_events(filtered, text_column, llm_client=llm_client)
    dissemination_input = enriched[[timestamp_column, text_column]].rename(columns={text_column: "text"})
    dissemination = compute_cluster_features(
        dissemination_input,
        text_column="text",
        timestamp_column=timestamp_column,
    )

    return {"filtered_news": enriched, "cluster_features": dissemination}


def save_outputs(
    outputs: Dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Persist pipeline outputs either as JSON or a pair of Parquet files."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        payload = {
            name: df.to_dict(orient="records") for name, df in outputs.items()
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        return

    filtered = outputs.get("filtered_news", pd.DataFrame())
    filtered.to_parquet(output_path)

    clusters = outputs.get("cluster_features", pd.DataFrame())
    clusters.to_parquet(output_path.with_name(output_path.stem + "_clusters.parquet"))

