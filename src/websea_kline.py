"""Utilities for fetching Websea futures kline data."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://oapi.websea.com"
KLINE_ENDPOINT = "/v1/futures/kline"
DEFAULT_TIMEOUT = 10
MAX_PAGE_SIZE = 2000
RATE_LIMIT_PER_10S = 100


@dataclass(frozen=True)
class KlineRequest:
    symbol: str
    period: str
    start: Optional[int] = None
    end: Optional[int] = None
    size: int = MAX_PAGE_SIZE


class WebseaAPIError(RuntimeError):
    """Raised when the Websea API returns an error response."""


def _to_seconds(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp())


def fetch_kline_page(request: KlineRequest) -> List[dict]:
    """Fetch a single page of kline data."""
    params = {
        "symbol": request.symbol,
        "period": request.period,
        "size": request.size,
    }
    if request.start is not None:
        params["start"] = request.start
    if request.end is not None:
        params["end"] = request.end

    response = requests.get(
        f"{BASE_URL}{KLINE_ENDPOINT}", params=params, timeout=DEFAULT_TIMEOUT
    )
    payload = response.json()
    if payload.get("errno") != 0:
        raise WebseaAPIError(payload)

    return payload.get("result", {}).get("data", [])


def fetch_kline_history(
    symbol: str,
    period: str,
    start: datetime,
    end: datetime,
    size: int = MAX_PAGE_SIZE,
    rate_limit_per_10s: int = RATE_LIMIT_PER_10S,
) -> List[dict]:
    """Fetch kline data between start and end datetimes.

    Websea's API supports up to 2000 candles per request. This helper
    walks forward across the desired range while respecting the rate limit.
    """
    start_ts = _to_seconds(start)
    end_ts = _to_seconds(end)

    all_rows: List[dict] = []
    cursor = start_ts
    requests_sent = 0
    window_start = time.monotonic()

    while cursor < end_ts:
        batch = fetch_kline_page(
            KlineRequest(symbol=symbol, period=period, start=cursor, end=end_ts, size=size)
        )
        if not batch:
            break

        all_rows.extend(batch)

        ids = [int(row["id"]) for row in batch if "id" in row]
        if not ids:
            break

        cursor = max(ids) + 1

        requests_sent += 1
        if requests_sent >= rate_limit_per_10s:
            elapsed = time.monotonic() - window_start
            if elapsed < 10:
                time.sleep(10 - elapsed)
            window_start = time.monotonic()
            requests_sent = 0

    return all_rows


def klines_to_dataframe(klines: Iterable[dict]) -> pd.DataFrame:
    """Convert raw kline data to a DataFrame."""
    df = pd.DataFrame(list(klines))
    if df.empty:
        return df

    df = df.rename(
        columns={
            "id": "timestamp",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "amount": "amount",
            "vol": "volume",
        }
    )
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    return df
