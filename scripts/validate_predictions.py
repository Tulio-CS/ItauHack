"""CLI para validar previsões estruturadas usando dados de mercado."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

if __name__ == "__main__" and __package__ is None:  # pragma: no cover - bootstrap CLI
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.validation import (
    create_accuracy_table,
    evaluate_predictions,
    generate_accuracy_plot,
    generate_confusion_heatmap,
    generate_multi_news_plot,
    generate_price_availability_plot,
    generate_rolling_window_plot,
    load_structured_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/output/investing_news_structured.jsonl"),
        help="Arquivo JSONL com saídas estruturadas do LLM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/validation"),
        help="Diretório onde tabelas e gráficos serão salvos.",
    )
    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=0.01,
        help="Retorno absoluto abaixo do qual o movimento é considerado neutro.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nível de log (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=[1, 3, 5],
        help="Horizontes em dias para checar o retorno (padrão: 1 3 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {args.input}")

    records = load_structured_records(args.input)
    (
        detailed_df,
        summaries,
        confusion_df,
        availability_df,
        multi_news_df,
        rolling_windows_df,
    ) = evaluate_predictions(
        records,
        horizons=tuple(sorted(set(args.horizons))),
        neutral_threshold=args.neutral_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = args.output_dir / "detailed_results.parquet"
    summary_path = args.output_dir / "accuracy_summary.csv"
    confusion_path = args.output_dir / "confusion_matrix.csv"
    availability_path = args.output_dir / "price_availability.csv"
    multi_news_path = args.output_dir / "multi_news_summary.csv"
    multi_news_plot = args.output_dir / "multi_news_accuracy.png"
    rolling_window_path = args.output_dir / "rolling_window_summary.csv"
    rolling_window_plot = args.output_dir / "rolling_window_accuracy.png"

    if detailed_df.empty:
        logging.warning("Nenhum resultado salvo porque não houve dados avaliados.")
    else:
        try:
            detailed_df.to_parquet(detailed_path, index=False)
        except Exception as exc:
            logging.warning("Falha ao salvar Parquet (%s). Salvando CSV como fallback.", exc)
            detailed_path = detailed_path.with_suffix(".csv")
            detailed_df.to_csv(detailed_path, index=False)
        accuracy_df = create_accuracy_table(summaries)
        accuracy_df.to_csv(summary_path, index=False)
        confusion_df.to_csv(confusion_path, index=False)

        accuracy_plot = args.output_dir / "accuracy_by_horizon.png"
        heatmap_plot = args.output_dir / "confusion_heatmap.png"
        generate_accuracy_plot(accuracy_df, accuracy_plot)
        generate_confusion_heatmap(confusion_df, heatmap_plot)

        logging.info("Resultados detalhados: %s", detailed_path)
        logging.info("Resumo de acurácia: %s", summary_path)
        logging.info("Confusion matrix: %s", confusion_path)
        logging.info("Gráfico de acurácia: %s", accuracy_plot)
        logging.info("Mapa de calor: %s", heatmap_plot)

    if not availability_df.empty:
        availability_df.to_csv(availability_path, index=False)
        availability_plot = args.output_dir / "price_availability.png"
        generate_price_availability_plot(availability_df, availability_plot)
        logging.info("Disponibilidade de preços: %s", availability_path)
        logging.info("Gráfico de disponibilidade de preços: %s", availability_plot)
    else:
        logging.warning("Nenhuma tentativa de busca de preço registrada.")

    if not multi_news_df.empty:
        multi_news_df.to_csv(multi_news_path, index=False)
        generate_multi_news_plot(multi_news_df, multi_news_plot)
        logging.info("Resumo de múltiplas notícias no dia: %s", multi_news_path)
        logging.info("Gráfico de acurácia (múltiplas notícias): %s", multi_news_plot)
    else:
        logging.info("Nenhum dia com múltiplas notícias para registrar.")

    if not rolling_windows_df.empty:
        rolling_windows_df.to_csv(rolling_window_path, index=False)
        generate_rolling_window_plot(rolling_windows_df, rolling_window_plot)
        logging.info(
            "Resumo por janela de notícias: %s", rolling_window_path
        )
        logging.info(
            "Gráfico de acurácia por janela: %s", rolling_window_plot
        )
    else:
        logging.info("Nenhuma janela de notícias agregada disponível para análise.")


if __name__ == "__main__":
    main()

