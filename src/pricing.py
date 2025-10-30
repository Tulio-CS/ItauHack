"""Utilities for fetching market data and computing returns."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


def load_price_window(
    ticker: str,
    event_time: pd.Timestamp,
    window: int = 1,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download price history around ``event_time``.

    Parameters
    ----------
    ticker:
        Asset ticker in Yahoo Finance notation.
    event_time:
        Timestamp associated with the news article.
    window:
        Number of days before and after the event to include.
    interval:
        Yahoo Finance interval string (``1d`` by default).
    """

    start = (event_time - timedelta(days=window + 1)).date()
    end = (event_time + timedelta(days=window + 1)).date()
    history = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
    history.index = pd.to_datetime(history.index)
    return history


def get_close_prices(history: pd.DataFrame, event_time: pd.Timestamp, forward_days: int = 1) -> pd.DataFrame:
    """Return prices between the last close before the event and ``forward_days`` after."""

    history = history.sort_index()
    before = history.loc[history.index <= event_time]
    after = history.loc[history.index > event_time]
    if before.empty:
        before_price = history.head(1)
    else:
        before_price = before.tail(1)
    after_price = after.head(forward_days)
    return pd.concat([before_price, after_price])
