"""Executa o agente de trading Janus e gera relatórios."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.janus_agent import AgentDecision, JanusAgent, NewsSignal
from src.simple_plot import save_line_chart

DATA_DIR = Path("data/output")
REPORT_DIR = Path("reports/output")
LOG_DIR = REPORT_DIR / "logs"
AGG_JSONL = DATA_DIR / "investing_news_structured.jsonl"
RESULTS_CSV = REPORT_DIR / "janus_agent_results.csv"
SUMMARY_CSV = REPORT_DIR / "janus_agent_summary.csv"
TABLE_CSV = REPORT_DIR / "janus_agent_results_table.csv"
PLOT_PATH = REPORT_DIR / "janus_agent_cumulative_pnl.svg"
LOG_PATH = LOG_DIR / "janus_agent.log"


class AgentRunner:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("janus_agent_runner")

    def load_news(self) -> List[Dict[str, object]]:
        if not AGG_JSONL.exists():
            raise FileNotFoundError("Dataset agregado não encontrado. Execute build_news_sentiment.py antes.")
        records: List[Dict[str, object]] = []
        with AGG_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    self.logger.warning("Linha inválida ignorada: %s", line[:80])
        return records

    def run(self) -> None:
        self.logger.info("Carregando dados agregados de notícias")
        news_records = self.load_news()
        agent = JanusAgent()
        signals = agent.load_news_signals(news_records)
        decisions = agent.run(signals)
        if not decisions:
            self.logger.warning("Nenhuma decisão gerada pelo agente")
            return
        self.logger.info("Gerando relatórios do agente")
        self._save_results(decisions)
        self.logger.info("Relatórios concluídos")

    def _save_results(self, decisions: List[AgentDecision]) -> None:
        decisions = list(decisions)
        with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "ticker",
                    "date",
                    "sentiment_score",
                    "spread_score",
                    "position",
                    "action",
                    "pnl",
                    "cumulative_pnl",
                ]
            )
            for decision in decisions:  # type: ignore[attr-defined]
                writer.writerow(
                    [
                        decision.ticker,
                        decision.date.date().isoformat(),
                        f"{decision.sentiment_score:.6f}",
                        f"{decision.spread_score:.6f}",
                        f"{decision.position:.6f}",
                        decision.action,
                        f"{decision.pnl:.6f}",
                        f"{decision.cumulative_pnl:.6f}",
                    ]
                )

        by_ticker: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "pnl": 0.0,
            "trades": 0,
        })
        timeline: Dict[date, float] = defaultdict(float)
        for decision in decisions:  # type: ignore[attr-defined]
            bucket = by_ticker[decision.ticker]
            bucket["pnl"] += decision.pnl
            bucket["trades"] += 1
            timeline[decision.date.date()] += decision.pnl

        with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ticker", "total_pnl", "num_trades"])
            for ticker, payload in sorted(by_ticker.items()):
                writer.writerow([ticker, f"{payload['pnl']:.6f}", payload["trades"]])

        with TABLE_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "daily_pnl", "cumulative_pnl"])
            cumulative = 0.0
            cumulative_points: List[Tuple[float, float]] = []
            for idx, (date, pnl) in enumerate(sorted(timeline.items())):
                cumulative += pnl
                writer.writerow([date.isoformat(), f"{pnl:.6f}", f"{cumulative:.6f}"])
                cumulative_points.append((float(idx), cumulative))

        if cumulative_points:
            save_line_chart(
                PLOT_PATH,
                cumulative_points,
                title="PnL acumulado do agente Janus",
                y_label="PnL",
            )
        self.logger.info(
            "Resultados salvos em %s, %s e %s",
            RESULTS_CSV,
            SUMMARY_CSV,
            PLOT_PATH,
        )


if __name__ == "__main__":
    AgentRunner().run()
