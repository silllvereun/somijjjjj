#!/usr/bin/env python3
"""Run DCA/martingale simulations on kline data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dca_simulator import run_all_simulations


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DCA simulations")
    parser.add_argument("--data", required=True, help="Path to CSV/XLSX with dt/close columns")
    parser.add_argument("--initial", type=float, default=10000, help="Initial capital")
    parser.add_argument("--coin", default="ETH", help="Coin type for min order sizing")
    args = parser.parse_args()

    df = load_data(Path(args.data))
    df["dt"] = pd.to_datetime(df["dt"])

    print("=" * 80)
    print("Simulation start")
    print("=" * 80)
    print(f"Initial capital: {args.initial:,.2f} USDT")
    print(f"Data range: {df['dt'].iloc[0]} ~ {df['dt'].iloc[-1]}")
    print(f"Rows: {len(df)}")
    print()

    results = run_all_simulations(df, initial_capital=args.initial, coin_type=args.coin)

    print("\n" + "=" * 80)
    print("Simulation summary")
    print("=" * 80)

    for logic_name, data in results.items():
        stats = data["stats"]
        sim = data["simulator"]
        trades_df = pd.DataFrame(sim.trades)
        entry_count = len(trades_df[trades_df["type"] == "ENTRY"]) if not trades_df.empty else 0
        exit_count = stats["total_trades"] - entry_count

        print(f"\n【{logic_name}】")
        print(f"  Initial capital: {stats['initial_capital']:,.0f} USDT")
        print(f"  Final capital: {stats['final_capital']:,.2f} USDT")
        print(f"  Net PnL: {stats['final_capital'] - stats['initial_capital']:+,.2f} USDT")
        print(f"  Total return: {stats['total_return']:+.2f}%")
        print()
        print("  Trades:")
        print(f"    - Entries: {entry_count}x")
        print(f"    - Exits: {exit_count}x")
        print(f"    - Total: {stats['total_trades']}x")
        print()
        print(f"  Win rate: {stats['win_rate']:.2f}%")
        print(f"  Max drawdown: {stats['max_drawdown']:.2f}%")
        print(f"  Profit factor: {stats['profit_factor']:.2f}")
        print(f"  Sharpe ratio: {stats['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()
