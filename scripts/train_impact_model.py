"""Treina um classificador XGBoost para prever o impacto das notícias."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from news_analysis.llm_interface import RuleBasedLLMClient
from news_analysis.modeling import (
    TrainConfig,
    make_binary_target,
    persist_metrics,
    prepare_dataset,
    train_xgboost_classifier,
)
from news_analysis.pipeline import run_pipeline

try:  # pragma: no cover - optional dependency
    import joblib
except Exception:  # pragma: no cover - fallback when joblib is ausente
    joblib = None  # type: ignore[assignment]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina e avalia um modelo de impacto baseado em notícias."
    )
    parser.add_argument("input_path", type=Path, help="Arquivo de notícias (CSV/Parquet).")
    parser.add_argument(
        "--text-column",
        default="headline",
        help="Coluna com o texto da notícia.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="created_at",
        help="Coluna com o timestamp da notícia.",
    )
    parser.add_argument(
        "--label-column",
        default="impact_1d",
        help="Coluna usada como rótulo para o modelo (ex.: impact_1d ou label).",
    )
    parser.add_argument(
        "--binary-target",
        action="store_true",
        help="Converte o rótulo numérico em 0/1 utilizando o limiar informado.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Limiar para definir o impacto positivo quando --binary-target estiver ativo.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["rule", "external"],
        default="rule",
        help="Escolhe o cliente LLM: heurístico local ou implementação externa.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Arquivo .joblib para salvar o modelo treinado.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/model_metrics.json"),
        help="Arquivo JSON com as métricas de avaliação.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Arquivo JSON com hiperparâmetros para o XGBoost.",
    )
    return parser


def load_config(path: Optional[Path]) -> TrainConfig:
    if not path:
        return TrainConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainConfig(**data)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    llm_client = None if args.llm_mode == "external" else RuleBasedLLMClient()

    outputs = run_pipeline(
        args.input_path,
        text_column=args.text_column,
        timestamp_column=args.timestamp_column,
        llm_client=llm_client,
    )

    enriched = outputs["filtered_news"]
    clusters = outputs["cluster_features"]

    if enriched.empty:
        raise SystemExit("Nenhuma notícia relevante encontrada para treinar o modelo.")

    if args.label_column not in enriched:
        raise SystemExit(
            f"Coluna de rótulo '{args.label_column}' não está disponível após o processamento."
        )

    X, y = prepare_dataset(
        enriched,
        clusters,
        label_column=args.label_column,
        timestamp_column=args.timestamp_column,
        drop_columns=[args.text_column, "news_category", "justificativa_impacto"],
    )

    if args.binary_target:
        y_binary = make_binary_target(y, threshold=args.threshold)
        common_index = y_binary.index.intersection(X.index)
        X = X.loc[common_index]
        y = y_binary.loc[common_index]
    elif not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes
        X = X.loc[y.index]

    if y.nunique() < 2:
        raise SystemExit(
            "O conjunto de treinamento possui apenas uma classe após o pré-processamento."
        )

    config = load_config(args.config)
    model, metrics = train_xgboost_classifier(X, y, config=config)

    persist_metrics(metrics, args.metrics_output)
    print(json.dumps({k: v for k, v in metrics.items() if k in {"accuracy", "roc_auc"}}, indent=2))

    if args.model_output:
        if joblib is None:
            raise SystemExit(
                "joblib não está disponível para serializar o modelo. Instale 'joblib'."
            )
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_output)


if __name__ == "__main__":
    main(sys.argv[1:])
