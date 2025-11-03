"""Executa o pipeline completo de integração Janus."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()

from src.agents.janus_trading_agent import AgentConfig, JanusTradingAgent
from src.pipelines.janus_news_pipeline import (
    aggregate_daily_sentiment,
    aggregate_overall_sentiment,
    copy_structured_file,
    export_structured_events,
    load_llm_csv_events,
    load_master_summary,
    load_structured_news,
    merge_sentiment_with_master,
    save_daily_sentiment,
    save_overall_sentiment,
)
from src.utils.png import save_line_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=Path("data/output/aapl_structured.jsonl"),
        help="Arquivo de origem com notícias estruturadas.",
    )
    parser.add_argument(
        "--target-jsonl",
        type=Path,
        default=Path("data/output/investing_news_structured.jsonl"),
        help="Destino consolidado das notícias estruturadas.",
    )
    parser.add_argument(
        "--news-csv",
        type=Path,
        default=Path("data/news_impact_labels_AAPL.csv"),
        help="Fallback com os rótulos de impacto avaliados pela LLM.",
    )
    parser.add_argument(
        "--master-summary",
        type=Path,
        default=Path("reports/output/summary_backtest.csv"),
        help="Resumo consolidado de métricas de mercado (proxy do master).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/news"),
        help="Diretório base para relatórios.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Nível de log (INFO, DEBUG, ...).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("janus_pipeline")

    events = []
    try:
        copy_structured_file(args.source_jsonl, args.target_jsonl)
        events = load_structured_news(args.target_jsonl)
    except FileNotFoundError:
        logger.warning(
            "Arquivo JSONL %s ausente; tentando construir a partir do CSV %s",
            args.source_jsonl,
            args.news_csv,
        )
    except ValueError as exc:
        logger.warning("Falha ao carregar JSONL estruturado: %s", exc)

    if not events:
        events = load_llm_csv_events(args.news_csv)
        if not events:
            logger.error(
                "Nenhum evento encontrado no CSV %s. Abortando pipeline.",
                args.news_csv,
            )
            return 1
        export_structured_events(args.target_jsonl, events)


    daily = aggregate_daily_sentiment(events)
    overall = aggregate_overall_sentiment(events)

    sentiment_dir = args.output_dir / "sentiment"
    save_daily_sentiment(sentiment_dir / "daily_sentiment.csv", daily)
    save_overall_sentiment(sentiment_dir / "overall_sentiment.csv", overall)

    master = load_master_summary(args.master_summary)
    merged_path = args.output_dir / "janus_sentiment_master.csv"
    merge_sentiment_with_master(daily, master, merged_path)

    agent = JanusTradingAgent(AgentConfig(sentiment_threshold=0.15, base_risk=1.5, max_risk=4.0))
    daily_rows = [
        {
            "ticker": item.ticker,
            "date": item.date,
            "mean_score": f"{item.mean_score:.4f}",
        }
        for item in daily
    ]
    history = agent.run(daily_rows, master)
    report = agent.summarise(history)

    agent_dir = args.output_dir / "agent"
    JanusTradingAgent.save_history_csv(agent_dir / "janus_agent_history.csv", history)
    JanusTradingAgent.save_report(agent_dir / "janus_agent_report.csv", report)

    equity_curve = [state.equity for state in history]
    save_line_plot(agent_dir / "janus_agent_equity.png", equity_curve)

    with (agent_dir / "janus_agent.log").open("w", encoding="utf-8") as handle:
        handle.write("Janus Trading Agent Log\n")
        handle.write(f"Config: {agent.config}\n")
        handle.write(f"Trades executados: {len(history)}\n")
        handle.write(
            f"Resumo -> Total PnL: {report.total_pnl:.4f}, Sharpe aproximado: {report.sharpe:.4f}, Média diária: {report.mean_daily_return:.4f}\n"
        )
        handle.write("\nÚltimos movimentos:\n")
        for state in history[-10:]:
            handle.write(
                f"{state.date} {state.ticker} | score={state.sentiment_score:.4f} | action={state.action} | pnl={state.realized_return:.4f} | equity={state.equity:.4f}\n"
            )

    logger.info("Pipeline concluído. Arquivos em %s", args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
