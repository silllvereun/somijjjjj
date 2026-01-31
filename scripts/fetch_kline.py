#!/usr/bin/env python3
"""Download Websea kline data and save to CSV."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.websea_kline import fetch_kline_history, klines_to_dataframe


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Websea futures kline data")
    parser.add_argument("--symbol", required=True, help="Symbol like BTC-USDT")
    parser.add_argument("--period", default="1min", help="Kline interval, e.g. 1min")
    parser.add_argument("--start", type=parse_datetime, required=True, help="Start ISO datetime")
    parser.add_argument("--end", type=parse_datetime, required=True, help="End ISO datetime")
    parser.add_argument("--output", default="kline.csv", help="Output CSV file path")
    args = parser.parse_args()

    klines = fetch_kline_history(
        symbol=args.symbol,
        period=args.period,
        start=args.start,
        end=args.end,
    )
    df = klines_to_dataframe(klines)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
