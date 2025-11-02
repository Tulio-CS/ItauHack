"""Agente simplificado de trading para o robô Janus.

O objetivo é combinar o spread teórico (proxy) com o sentimento
agregado das notícias do dia, produzindo uma trajetória de PnL e logs
interpretáveis. O módulo usa somente recursos da stdlib para manter a
compatibilidade com ambientes sem acesso à internet.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    sentiment_threshold: float = 0.1
    base_risk: float = 1.0
    max_risk: float = 3.0


@dataclass
class DailyState:
    date: str
    ticker: str
    sentiment_score: float
    action: str
    position_size: float
    expected_return: float
    realized_return: float
    equity: float


@dataclass
class AgentReport:
    total_pnl: float
    mean_daily_return: float
    stdev_daily_return: float
    sharpe: float
    trades: int


def _safe_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


class JanusTradingAgent:
    """Agente baseado em regras que mistura spread e sentimento."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()

    def _position_from_sentiment(self, score: float) -> float:
        magnitude = abs(score)
        if magnitude < self.config.sentiment_threshold:
            return 0.0
        scaled = self.config.base_risk * magnitude / max(self.config.sentiment_threshold, 1e-6)
        return max(-self.config.max_risk, min(self.config.max_risk, math.copysign(scaled, score)))

    def run(
        self,
        daily_sentiment: Iterable[Dict[str, str]],
        master_summary: Dict[str, Dict[str, str]],
    ) -> List[DailyState]:
        equity = 0.0
        history: List[DailyState] = []
        for row in daily_sentiment:
            ticker = row["ticker"].strip().upper()
            base = master_summary.get(ticker)
            if not base:
                logger.debug("Ticker %s não encontrado no master", ticker)
                continue

            score = _safe_float(row["mean_score"])
            position = self._position_from_sentiment(score)
            if position == 0.0:
                action = "flat"
            elif position > 0:
                action = "long_bdr_short_adr"
            else:
                action = "short_bdr_long_adr"

            pnl_per_trade = _safe_float(base.get("pnl_base")) / max(_safe_float(base.get("trades_base"), 1.0), 1.0)
            expected = position * pnl_per_trade
            realized = expected * (1.0 - abs(_safe_float(base.get("median_ratio_diag"), 0.0)))
            equity += realized

            history.append(
                DailyState(
                    date=row["date"],
                    ticker=ticker,
                    sentiment_score=score,
                    action=action,
                    position_size=round(position, 4),
                    expected_return=round(expected, 4),
                    realized_return=round(realized, 4),
                    equity=round(equity, 4),
                )
            )
        return history

    @staticmethod
    def summarise(history: Iterable[DailyState]) -> AgentReport:
        returns = [state.realized_return for state in history]
        total = sum(returns)
        count = len(returns)
        mean_return = total / count if count else 0.0
        variance = (
            sum((value - mean_return) ** 2 for value in returns) / (count - 1)
            if count > 1
            else 0.0
        )
        stdev = math.sqrt(variance)
        sharpe = mean_return / stdev if stdev else 0.0
        return AgentReport(
            total_pnl=round(total, 4),
            mean_daily_return=round(mean_return, 4),
            stdev_daily_return=round(stdev, 4),
            sharpe=round(sharpe, 4),
            trades=count,
        )

    @staticmethod
    def save_history_csv(path: Path, history: Iterable[DailyState]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "date",
            "ticker",
            "sentiment_score",
            "action",
            "position_size",
            "expected_return",
            "realized_return",
            "equity",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for state in history:
                writer.writerow({field: getattr(state, field) for field in fieldnames})

    @staticmethod
    def save_report(path: Path, report: AgentReport) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("total_pnl,mean_daily_return,stdev_daily_return,sharpe,trades\n")
            handle.write(
                f"{report.total_pnl:.4f},{report.mean_daily_return:.4f},{report.stdev_daily_return:.4f},{report.sharpe:.4f},{report.trades}\n"
            )
