"""Processamento completo de notícias para classificação, extração e clusterização."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from news_analysis.llm_interface import RuleBasedLLMClient
from news_analysis.pipeline import run_pipeline, save_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de enriquecimento de notícias financeiras."
    )
    parser.add_argument(
        "input_path", type=Path, help="Arquivo com notícias em CSV ou Parquet."
    )
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
        "--output",
        type=Path,
        default=Path("reports/news_features.parquet"),
        help="Arquivo de saída (Parquet ou JSON).",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["rule", "external"],
        default="rule",
        help="Escolhe o cliente LLM: heurístico local ou implementação externa.",
    )
    return parser


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

    save_outputs(outputs, args.output)
    print("Processamento concluído.")


if __name__ == "__main__":
    main(sys.argv[1:])
