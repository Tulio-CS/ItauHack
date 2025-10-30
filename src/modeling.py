"""Model training and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass
class TrainingResult:
    model: XGBClassifier
    report: Dict[str, Dict[str, float]]
    confusion: np.ndarray
    feature_names: Tuple[str, ...]


def train_model(features: pd.DataFrame, labels: pd.Series, random_state: int = 42) -> TrainingResult:
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test, predictions)
    return TrainingResult(model=model, report=report, confusion=confusion, feature_names=tuple(features.columns))


def save_model(result: TrainingResult, path: Path) -> None:
    payload = {
        "model": result.model,
        "feature_names": result.feature_names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_model(path: Path) -> TrainingResult:
    payload = joblib.load(path)
    model = payload["model"]
    feature_names = tuple(payload["feature_names"])
    return TrainingResult(model=model, report={}, confusion=np.array([]), feature_names=feature_names)
