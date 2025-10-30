"""Command line entry-point to run the complete news analysis workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.features import build_feature_matrix, compute_price_impact, compute_return_label
from src.modeling import save_model, train_model
from src.news_processing import NewsAnalyzer, combine_news_frames
from src.pricing import get_close_prices, load_price_window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM assisted news impact analysis")
    parser.add_argument(
        "--news-files",
        nargs="+",
        required=True,
        help="Paths to parquet files with Investing.com news",
    )
    parser.add_argument(
        "--text-column",
        default="headline",
        help="Column containing the news headline/text",
    )
    parser.add_argument(
        "--datetime-column",
        default="datetime",
        help="Column with the publication datetime in ISO format",
    )
    parser.add_argument(
        "--ticker-column",
        default="ticker",
        help="Column containing the asset ticker symbol",
    )
    parser.add_argument(
        "--default-ticker",
        default="AAPL",
        help="Fallback ticker used when the column is missing",
    )
    parser.add_argument(
        "--forward-days",
        type=int,
        default=1,
        help="Number of days after the event used to compute returns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated"),
        help="Directory to store tables, charts and the trained model",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("reports/generated/models/news_xgb.joblib"),
        help="Path where the trained model will be stored",
    )
    return parser.parse_args()


def load_news_frames(paths: List[str]) -> List[pd.DataFrame]:
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        frames.append(df)
    return frames


def enrich_with_market_data(
    df: pd.DataFrame,
    datetime_column: str,
    ticker_column: str,
    forward_days: int,
    default_ticker: str,
) -> pd.DataFrame:
    returns = []
    for row in df.itertuples(index=False):
        event_time = pd.to_datetime(getattr(row, datetime_column))
        ticker = getattr(row, ticker_column, None) or default_ticker
        prices = load_price_window(ticker, event_time, window=forward_days)
        close_prices = get_close_prices(prices, event_time, forward_days=forward_days)
        impact = compute_price_impact(close_prices)
        label = compute_return_label(impact)
        returns.append({
            "retorno_forward": impact,
            "impact_label": label,
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(returns)], axis=1)


def plot_impact_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    df["retorno_forward"].hist(bins=30, ax=ax)
    ax.set_title("Distribuição do retorno após notícias market-moving")
    ax.set_xlabel("Retorno (% do preço)")
    ax.set_ylabel("Frequência")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "retorno_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(result, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(result.confusion)
    display.plot(ax=ax)
    ax.set_title("Matriz de confusão (impacto preço)")
    path = output_dir / "confusao_modelo.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(result, output_dir: Path) -> Path:
    importance = result.model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    order = importance.argsort()[::-1]
    ax.barh(range(len(importance)), importance[order])
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([result.feature_names[i] for i in order])
    ax.set_title("Importância das features (XGBoost)")
    fig.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_news_frames(args.news_files)
    raw_news = combine_news_frames(frames)

    analyzer = NewsAnalyzer()
    processed = analyzer.process_dataframe(raw_news, text_column=args.text_column)
    processed.to_parquet(output_dir / "market_moving_news.parquet", index=False)

    enriched = enrich_with_market_data(
        processed,
        datetime_column=args.datetime_column,
        ticker_column=args.ticker_column,
        forward_days=args.forward_days,
        default_ticker=args.default_ticker,
    )
    enriched.to_parquet(output_dir / "market_moving_with_returns.parquet", index=False)

    features = build_feature_matrix(enriched)
    features.to_parquet(output_dir / "features.parquet", index=False)

    labels = enriched["impact_label"]
    result = train_model(features, labels)
    save_model(result, args.model_path)

    report_path = output_dir / "classification_report.json"
    report_path.write_text(json.dumps(result.report, indent=2, ensure_ascii=False))

    hist_path = plot_impact_distribution(enriched, output_dir)
    confusion_path = plot_confusion(result, output_dir)
    importance_path = plot_feature_importance(result, output_dir)

    summary = {
        "n_raw_news": int(len(raw_news)),
        "n_market_moving": int(len(processed)),
        "histogram": str(hist_path),
        "confusion": str(confusion_path),
        "importance": str(importance_path),
        "model_path": str(args.model_path),
        "report_path": str(report_path),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
