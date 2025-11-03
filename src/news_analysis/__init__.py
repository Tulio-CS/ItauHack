"""Utilities for filtering, extracting and featurizing financial news."""

from .relevance_filter import NewsCategory, NewsRelevanceFilter
from .event_extractor import StructuredEventExtractor, StructuredEvent
from .cluster_features import ClusterFeatureEngineer
from .llm_interface import LLMClient, RuleBasedLLMClient
from .pipeline import (
    compute_cluster_features,
    extract_events,
    load_news,
    relevance_filter,
    run_pipeline,
    save_outputs,
)
from .modeling import (
    TrainConfig,
    make_binary_target,
    persist_metrics,
    prepare_dataset,
    train_xgboost_classifier,
)

__all__ = [
    "NewsCategory",
    "NewsRelevanceFilter",
    "StructuredEventExtractor",
    "StructuredEvent",
    "ClusterFeatureEngineer",
    "LLMClient",
    "RuleBasedLLMClient",
    "compute_cluster_features",
    "extract_events",
    "load_news",
    "relevance_filter",
    "run_pipeline",
    "save_outputs",
    "TrainConfig",
    "make_binary_target",
    "persist_metrics",
    "prepare_dataset",
    "train_xgboost_classifier",
]
