"""Helpers to assemble feature matrices and treinar modelos de impacto com XGBoost."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - handled at runtime
    XGBClassifier = None  # type: ignore[misc]


@dataclass
class TrainConfig:
    """Configuração padrão para o classificador de impacto."""

    test_size: float = 0.2
    random_state: int = 42
    learning_rate: float = 0.1
    max_depth: int = 4
    n_estimators: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8


def _prepare_merge(
    enriched: pd.DataFrame,
    clusters: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """Merge news-level features with dissemination signals aligned by timestamp."""

    enriched = enriched.copy()
    enriched[timestamp_column] = pd.to_datetime(enriched[timestamp_column])

    clusters = clusters.copy()
    if not clusters.empty:
        clusters["window_end"] = pd.to_datetime(clusters["window_end"])
    else:
        clusters["window_end"] = pd.Series(dtype="datetime64[ns]")

    merged = pd.merge_asof(
        enriched.sort_values(timestamp_column),
        clusters.sort_values("window_end"),
        left_on=timestamp_column,
        right_on="window_end",
        direction="backward",
    )

    merged.drop(columns=["window_start"], inplace=True, errors="ignore")
    return merged


def _split_features_targets(
    dataset: pd.DataFrame,
    label_column: str,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a feature matrix (X) and the target vector (y)."""

    drop_columns = drop_columns or []

    non_feature_cols = set(drop_columns + [label_column])
    candidate_cols = [col for col in dataset.columns if col not in non_feature_cols]

    structured_cols = [
        col
        for col in candidate_cols
        if dataset[col].apply(lambda x: isinstance(x, (dict, list))).any()
    ]

    feature_df = dataset.drop(columns=structured_cols + list(non_feature_cols), errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col
        for col in feature_df.columns
        if col not in numeric_cols and feature_df[col].dtype == "object"
    ]

    X_parts: List[pd.DataFrame] = []
    if numeric_cols:
        X_parts.append(feature_df[numeric_cols].fillna(0.0))

    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(feature_df[categorical_cols].fillna("desconhecido"))
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        X_parts.append(pd.DataFrame(encoded, columns=encoded_cols, index=feature_df.index))

    if not X_parts:
        raise ValueError("Nenhuma feature disponível após o pré-processamento.")

    X = pd.concat(X_parts, axis=1)
    y = dataset[label_column]
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


def prepare_dataset(
    enriched: pd.DataFrame,
    clusters: pd.DataFrame,
    label_column: str,
    timestamp_column: str,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Combine e preparar o dataset final para modelagem."""

    merged = _prepare_merge(enriched, clusters, timestamp_column)
    drop_columns = drop_columns or []
    drop_columns = list(
        set(drop_columns + [timestamp_column, "structured_event", "window_end"])
    )

    X, y = _split_features_targets(merged, label_column=label_column, drop_columns=drop_columns)
    return X, y


def make_binary_target(series: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Transform numeric impacts into uma variável binária (>= limiar é 1)."""

    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.notna()
    return (numeric[mask] >= threshold).astype(int)


def train_xgboost_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[TrainConfig] = None,
) -> Tuple[object, Dict[str, object]]:
    """Train an XGBoost classifier and compute avaliação básica."""

    if XGBClassifier is None:  # pragma: no cover - runtime guard
        raise ImportError(
            "xgboost não está instalado. Instale 'xgboost' para treinar o modelo."
        )

    config = config or TrainConfig()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y if len(y.unique()) > 1 else None,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        n_estimators=config.n_estimators,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        random_state=config.random_state,
        use_label_encoder=False,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(
            y_test, preds, digits=4, output_dict=True
        ),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    except ValueError:
        metrics["roc_auc"] = None

    metrics["y_test"] = y_test.reset_index(drop=True)
    metrics["y_pred"] = pd.Series(preds)
    metrics["y_proba"] = pd.Series(proba)

    return model, metrics


def persist_metrics(metrics: Dict[str, object], output_path: Path) -> None:
    """Save evaluation metrics em JSON para fácil inspeção posterior."""

    serializable: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, pd.Series):
            serializable[key] = value.tolist()
        elif isinstance(value, dict):
            serializable[key] = value
        else:
            serializable[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable, indent=2, default=float), encoding="utf-8")

