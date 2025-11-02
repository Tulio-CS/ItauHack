"""Implementação simplificada do agente Janus.

O agente combina o spread estático (derivado dos relatórios de backtest) com o
sentimento agregado das notícias para definir posição e retorno simulado. A
lógica é determinística e livre de dependências externas para facilitar a
reprodutibilidade neste ambiente restrito.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class NewsSignal:
    ticker: str
    date: datetime
    mean_sentiment: float
    news_count: int
    spread_score: float | None
    sharpe_kf: float | None


@dataclass
class AgentDecision:
    ticker: str
    date: datetime
    sentiment_score: float
    spread_score: float
    position: float
    action: str
    pnl: float
    cumulative_pnl: float


class JanusAgent:
    """Agente determinístico que converte sentimento + spread em decisões."""

    def __init__(self, volatility_target: float = 0.08, spread_inertia: float = 0.6) -> None:
        self.volatility_target = volatility_target
        self.spread_inertia = spread_inertia

    @staticmethod
    def load_news_signals(records: Iterable[Dict[str, object]]) -> List[NewsSignal]:
        signals: List[NewsSignal] = []
        for entry in records:
            ticker = str(entry.get("ticker", "")).strip()
            if not ticker:
                continue
            date_str = str(entry.get("date", "")).strip()
            try:
                date = datetime.fromisoformat(date_str)
            except ValueError:
                continue
            try:
                mean_sentiment = float(entry.get("mean_sentiment", 0.0))
            except (TypeError, ValueError):
                mean_sentiment = 0.0
            try:
                news_count = int(entry.get("news_count", 0))
            except (TypeError, ValueError):
                news_count = 0
            spread_score = entry.get("spread_score")
            if spread_score is not None:
                try:
                    spread_score = float(spread_score)
                except (TypeError, ValueError):
                    spread_score = None
            sharpe_kf = entry.get("sharpe_kf")
            if sharpe_kf is not None:
                try:
                    sharpe_kf = float(sharpe_kf)
                except (TypeError, ValueError):
                    sharpe_kf = None
            signals.append(
                NewsSignal(
                    ticker=ticker,
                    date=date,
                    mean_sentiment=mean_sentiment,
                    news_count=news_count,
                    spread_score=spread_score,
                    sharpe_kf=sharpe_kf,
                )
            )
        signals.sort(key=lambda item: (item.ticker, item.date))
        return signals

    def _normalize_spread(self, signals: Sequence[NewsSignal]) -> Dict[str, float]:
        max_abs = 0.0
        for signal in signals:
            if signal.spread_score is not None:
                max_abs = max(max_abs, abs(signal.spread_score))
        if max_abs == 0:
            return {}
        return {
            signal.ticker: (signal.spread_score or 0.0) / max_abs for signal in signals
        }

    def _derive_sentiment_weight(self, signal: NewsSignal) -> float:
        base = signal.mean_sentiment
        intensity = min(signal.news_count / 5.0, 1.0)
        return base * (0.5 + 0.5 * intensity)

    def run(self, signals: Sequence[NewsSignal]) -> List[AgentDecision]:
        if not signals:
            return []
        spread_reference = self._normalize_spread(signals)
        cumulative: Dict[str, float] = {}
        results: List[AgentDecision] = []
        for signal in signals:
            spread_signal = spread_reference.get(signal.ticker, 0.0)
            if signal.spread_score is None:
                spread_signal *= 0.5
            sentiment_weight = self._derive_sentiment_weight(signal)
            base_position = sentiment_weight * (self.spread_inertia + (1 - self.spread_inertia) * (signal.sharpe_kf or 0.0))
            position = max(min(base_position, 1.5), -1.5)
            action = "comprado" if position > 0.05 else "vendido" if position < -0.05 else "neutro"
            effective_vol = self.volatility_target * (1 + abs(spread_signal))
            pnl = position * spread_signal * effective_vol
            cumulative_pnl = cumulative.get(signal.ticker, 0.0) + pnl
            cumulative[signal.ticker] = cumulative_pnl
            results.append(
                AgentDecision(
                    ticker=signal.ticker,
                    date=signal.date,
                    sentiment_score=sentiment_weight,
                    spread_score=spread_signal,
                    position=position,
                    action=action,
                    pnl=pnl,
                    cumulative_pnl=cumulative_pnl,
                )
            )
        return results


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))  # type: ignore[name-defined]
            except json.JSONDecodeError:
                continue
    return records
