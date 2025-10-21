"""Processamento completo de notícias para classificação, extração e clusterização."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from news_analysis import (
    ClusterFeatureEngineer,
    NewsCategory,
    NewsRelevanceFilter,
    RuleBasedLLMClient,
    StructuredEventExtractor,
)


def load_news(path: Path, text_column: str = "headline", timestamp_column: str = "created_at") -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Formato de arquivo não suportado. Use CSV ou Parquet.")

    if text_column not in df:
        raise ValueError(f"Coluna '{text_column}' não encontrada no arquivo {path}.")
    if timestamp_column not in df:
        raise ValueError(f"Coluna '{timestamp_column}' não encontrada no arquivo {path}.")
    return df


def relevance_filter(df: pd.DataFrame, text_column: str, llm_client=None) -> pd.DataFrame:
    llm_client = llm_client or RuleBasedLLMClient()
    filterer = NewsRelevanceFilter(llm_client)
    categories: List[NewsCategory] = []
    for text in df[text_column]:
        categories.append(filterer.classify(str(text)))
    df = df.copy()
    df["news_category"] = categories
    return df[df["news_category"] == NewsCategory.MARKET_MOVING]


def extract_events(df: pd.DataFrame, text_column: str, llm_client=None) -> pd.DataFrame:
    extractor = StructuredEventExtractor(llm_client)
    events = []
    event_types: List[str] = []
    sentiments: List[str] = []
    impact_scores: List[int | None] = []
    impact_rationales: List[str | None] = []
    metric_features: List[dict] = []
    for text in df[text_column]:
        event = extractor.extract(str(text))
        events.append(event.to_dict())
        event_types.append(event.event_type)
        sentiments.append(event.overall_sentiment)
        impact_scores.append(event.impact_rating)
        impact_rationales.append(event.impact_rationale)

        metrics_dict: dict[str, float | str | None] = {}
        for metric in event.metrics:
            prefix = metric.metric.lower().replace(" ", "_")
            if metric.value is not None:
                metrics_dict[f"{prefix}_valor"] = metric.value
            if metric.expectation is not None:
                metrics_dict[f"{prefix}_expectativa"] = metric.expectation
            if metric.outcome:
                metrics_dict[f"{prefix}_resultado"] = metric.outcome
        metric_features.append(metrics_dict)

    df = df.copy()
    df["structured_event"] = events
    df["evento_tipo"] = event_types
    df["sentimento_geral"] = sentiments
    df["impacto_potencial"] = impact_scores
    df["justificativa_impacto"] = impact_rationales

    metrics_df = pd.DataFrame(metric_features)
    df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)
    return df


def compute_cluster_features(df: pd.DataFrame, text_column: str, timestamp_column: str) -> pd.DataFrame:
    engineer = ClusterFeatureEngineer()
    return engineer.transform(df, text_column=text_column, timestamp_column=timestamp_column)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de enriquecimento de notícias financeiras.")
    parser.add_argument("input_path", type=Path, help="Arquivo com notícias em CSV ou Parquet.")
    parser.add_argument("--text-column", default="headline", help="Coluna com o texto da notícia.")
    parser.add_argument("--timestamp-column", default="created_at", help="Coluna com o timestamp da notícia.")
    parser.add_argument("--output", type=Path, default=Path("reports/news_features.parquet"))
    parser.add_argument("--llm-mode", choices=["rule", "external"], default="rule")
    args = parser.parse_args()

    llm_client = None if args.llm_mode == "external" else RuleBasedLLMClient()

    df = load_news(args.input_path, args.text_column, args.timestamp_column)
    filtered = relevance_filter(df, args.text_column, llm_client)
    enriched = extract_events(filtered, args.text_column, llm_client)
    dissemination_input = enriched[[args.timestamp_column, args.text_column]].rename(
        columns={args.text_column: "text"}
    )
    dissemination = compute_cluster_features(
        dissemination_input,
        text_column="text",
        timestamp_column=args.timestamp_column,
    )

    output = {
        "filtered_news": enriched,
        "cluster_features": dissemination,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".json":
        with args.output.open("w", encoding="utf-8") as fp:
            json.dump({key: df.to_dict(orient="records") for key, df in output.items()}, fp, indent=2, ensure_ascii=False)
    else:
        # salva como parquet multi-aba
        enriched.to_parquet(args.output)
        dissemination.to_parquet(args.output.with_name(args.output.stem + "_clusters.parquet"))

    print("Processamento concluído.")


if __name__ == "__main__":
    main()
