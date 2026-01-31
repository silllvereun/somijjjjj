"""Trading simulators for DCA/Martingale strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Entry:
    price: float
    quantity: float
    timestamp: datetime


class Position:
    """Position management helper."""

    def __init__(self, side: str, entry_price: float, quantity: float, leverage: int = 10):
        self.side = side
        self.entries: List[Entry] = []
        self.leverage = leverage
        self.add_entry(entry_price, quantity)

    def add_entry(self, price: float, quantity: float) -> None:
        self.entries.append(Entry(price=price, quantity=quantity, timestamp=datetime.now()))

    @property
    def total_quantity(self) -> float:
        return sum(e.quantity for e in self.entries)

    @property
    def avg_entry_price(self) -> float:
        total_cost = sum(e.price * e.quantity for e in self.entries)
        return total_cost / self.total_quantity if self.total_quantity > 0 else 0

    def get_roe(self, current_price: float) -> float:
        if self.total_quantity == 0:
            return 0

        avg_price = self.avg_entry_price
        if self.side == "LONG":
            pnl_percent = ((current_price - avg_price) / avg_price) * 100
        else:
            pnl_percent = ((avg_price - current_price) / avg_price) * 100

        return pnl_percent * self.leverage

    def get_unrealized_pnl(self, current_price: float, initial_capital: float) -> float:
        roe = self.get_roe(current_price)
        margin_used = self.get_margin_used(initial_capital)
        return margin_used * (roe / 100)

    def get_margin_used(self, initial_capital: float) -> float:
        total_notional = sum(e.price * e.quantity for e in self.entries)
        return total_notional / self.leverage

    def get_margin_ratio(self, initial_capital: float) -> float:
        return (self.get_margin_used(initial_capital) / initial_capital) * 100

    def close_partial(self, ratio: float, close_price: float, initial_capital: float) -> float:
        pnl = self.get_unrealized_pnl(close_price, initial_capital) * ratio
        for entry in self.entries:
            entry.quantity *= (1 - ratio)
        return pnl

    def close_all(self, close_price: float, initial_capital: float) -> float:
        pnl = self.get_unrealized_pnl(close_price, initial_capital)
        self.entries = []
        return pnl


class TradingSimulator:
    """Base class for strategy simulators."""

    def __init__(self, initial_capital: float = 10000, leverage: int = 10, fee_rate: float = 0.0004):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate

        self.position: Optional[Position] = None
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def calculate_entry_quantity(
        self,
        price: float,
        target_margin_ratio: float,
        max_entries: int,
        min_quantity: float,
    ) -> float:
        """Calculate entry quantity based on margin budget and minimum quantity."""
        if max_entries <= 0:
            return min_quantity

        total_margin_budget = self.initial_capital * (target_margin_ratio / 100)
        per_entry_margin = total_margin_budget / max_entries
        per_entry_notional = per_entry_margin * self.leverage
        quantity = per_entry_notional / price

        return max(quantity, min_quantity)

    def get_current_margin_ratio(self) -> float:
        if self.position is None:
            return 0
        return self.position.get_margin_ratio(self.initial_capital)

    def get_current_roe(self, current_price: float) -> float:
        if self.position is None:
            return 0
        return self.position.get_roe(current_price)

    def open_position(self, side: str, price: float, quantity: float, timestamp: Optional[datetime] = None) -> None:
        if self.position is None:
            self.position = Position(side, price, quantity, self.leverage)
        else:
            self.position.add_entry(price, quantity)

        fee = (price * quantity / self.leverage) * self.fee_rate
        self.capital -= fee

        if timestamp:
            self.trades.append(
                {
                    "timestamp": timestamp,
                    "type": "ENTRY",
                    "price": price,
                    "quantity": quantity,
                    "roe": 0,
                    "pnl": -fee,
                    "capital": self.capital,
                }
            )

    def close_position(self, price: float, partial_ratio: float = 1.0) -> float:
        if self.position is None:
            return 0

        if partial_ratio >= 1.0:
            pnl = self.position.close_all(price, self.initial_capital)
            notional = self.position.avg_entry_price * self.position.total_quantity
            fee = (notional / self.leverage) * self.fee_rate
            self.position = None
        else:
            pnl = self.position.close_partial(partial_ratio, price, self.initial_capital)
            notional = self.position.avg_entry_price * self.position.total_quantity * partial_ratio
            fee = (notional / self.leverage) * self.fee_rate

        net_pnl = pnl - fee
        self.capital += net_pnl

        return net_pnl

    def record_trade(
        self,
        trade_type: str,
        price: float,
        quantity: float,
        roe: float,
        pnl: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self.trades.append(
            {
                "timestamp": timestamp,
                "type": trade_type,
                "price": price,
                "quantity": quantity,
                "roe": roe,
                "pnl": pnl,
                "capital": self.capital,
            }
        )

    def record_equity(self, timestamp: datetime, price: float) -> None:
        unrealized_pnl = 0.0
        if self.position:
            unrealized_pnl = self.position.get_unrealized_pnl(price, self.initial_capital)

        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "price": price,
                "capital": self.capital,
                "unrealized_pnl": unrealized_pnl,
                "total_equity": self.capital + unrealized_pnl,
            }
        )

    def get_statistics(self) -> Dict:
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]

        equity_series = equity_df["total_equity"]
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24 * 60) if returns.std() > 0 else 0

        stats = {
            "initial_capital": self.initial_capital,
            "final_capital": self.capital,
            "total_return": ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0,
            "avg_win": winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0,
            "largest_win": trades_df["pnl"].max(),
            "largest_loss": trades_df["pnl"].min(),
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "profit_factor": (
                abs(winning_trades["pnl"].sum() / losing_trades["pnl"].sum())
                if len(losing_trades) > 0 and losing_trades["pnl"].sum() != 0
                else float("inf")
            ),
        }

        return stats


class Logic1Simulator(TradingSimulator):
    """Basic DCA strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_interval = 150
        self.max_entries = 576
        self.entry_count = 0
        self.last_entry_time: Optional[datetime] = None
        self.take_profit_roe = 2.05

    def run(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            current_time = row["dt"]
            current_price = row["close"]

            if self.position:
                roe = self.get_current_roe(current_price)

                if roe >= self.take_profit_roe:
                    pnl = self.close_position(current_price)
                    self.record_trade("TAKE_PROFIT", current_price, 0, roe, pnl, current_time)
                    self.entry_count = 0
                    self.last_entry_time = None
                elif self.entry_count >= self.max_entries:
                    pnl = self.close_position(current_price)
                    self.record_trade("MAX_ENTRIES", current_price, 0, roe, pnl, current_time)
                    self.entry_count = 0
                    self.last_entry_time = None

            if self.last_entry_time is None or (
                current_time - self.last_entry_time
            ).total_seconds() >= self.entry_interval:
                if self.entry_count < self.max_entries:
                    quantity = 20 / current_price
                    self.open_position("LONG", current_price, quantity, current_time)
                    self.entry_count += 1
                    self.last_entry_time = current_time

            self.record_equity(current_time, current_price)


class Logic2Simulator(TradingSimulator):
    """DCA with loss control and minimum quantity per coin."""

    def __init__(self, *args, coin_type: str = "ETH", **kwargs):
        super().__init__(*args, **kwargs)
        self.last_entry_time: Optional[datetime] = None
        self.entry_interval = 150
        self.max_entries = 576
        self.last_entry_roe = 0.0
        self.first_stop_triggered = False
        self.entry_prohibited = False
        self.take_profit_roe = 2.5
        self.dca_margin_ratio = 10
        self.coin_type = coin_type
        self.min_quantity = 0.001 if coin_type == "BTC" else 0.01 if coin_type == "ETH" else 0.1

    def run(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            current_time = row["dt"]
            current_price = row["close"]

            margin_ratio = self.get_current_margin_ratio()
            roe = self.get_current_roe(current_price)

            if roe < -10 or margin_ratio >= 30:
                self.record_equity(current_time, current_price)
                continue

            if self.position:
                if roe <= -15:
                    pnl = self.close_position(current_price)
                    self.record_trade("STOP_LOSS_2", current_price, 0, roe, pnl, current_time)
                    self.first_stop_triggered = False
                    self.last_entry_roe = 0
                    self.entry_prohibited = False
                elif roe <= -12 and not self.first_stop_triggered:
                    pnl = self.close_position(current_price, partial_ratio=0.5)
                    self.record_trade(
                        "STOP_LOSS_1",
                        current_price,
                        self.position.total_quantity if self.position else 0,
                        roe,
                        pnl,
                        current_time,
                    )
                    self.first_stop_triggered = True
                    self.entry_prohibited = True
                elif self.first_stop_triggered and roe >= -8:
                    self.entry_prohibited = False
                elif roe >= self.take_profit_roe:
                    pnl = self.close_position(current_price)
                    self.record_trade("TAKE_PROFIT", current_price, 0, roe, pnl, current_time)
                    self.first_stop_triggered = False
                    self.last_entry_roe = 0
                    self.entry_prohibited = False

            if not self.entry_prohibited:
                if margin_ratio < 10:
                    if self.last_entry_time is None or (
                        current_time - self.last_entry_time
                    ).total_seconds() >= self.entry_interval:
                        quantity = self.calculate_entry_quantity(
                            price=current_price,
                            target_margin_ratio=self.dca_margin_ratio,
                            max_entries=self.max_entries,
                            min_quantity=self.min_quantity,
                        )
                        self.open_position("LONG", current_price, quantity, current_time)
                        self.last_entry_time = current_time
                        self.last_entry_roe = roe
                elif margin_ratio >= 10:
                    if self.last_entry_roe == 0 and roe <= -0.1:
                        self.open_position("LONG", current_price, self.min_quantity, current_time)
                        self.last_entry_roe = roe
                    elif self.last_entry_roe != 0 and roe <= self.last_entry_roe - 0.5:
                        self.open_position("LONG", current_price, self.min_quantity, current_time)
                        self.last_entry_roe = roe

            self.record_equity(current_time, current_price)


class Logic3Simulator(TradingSimulator):
    """1000-split DCA with reversal martingale."""

    def __init__(self, *args, coin_type: str = "ETH", **kwargs):
        super().__init__(*args, **kwargs)
        self.dca_phase = True
        self.dca_count = 0
        self.max_dca_entries = 1000
        self.entry_interval = 60
        self.last_entry_time: Optional[datetime] = None
        self.martingale_step = 0
        self.position_direction = "LONG"
        self.take_profit_roe = 5.0
        self.reduced_tp = False
        self.dca_stopped = False
        self.dca_margin_ratio = 5
        self.coin_type = coin_type
        self.min_quantity = 0.001 if coin_type == "BTC" else 0.01 if coin_type == "ETH" else 0.1

    def run(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            current_time = row["dt"]
            current_price = row["close"]

            margin_ratio = self.get_current_margin_ratio()
            roe = self.get_current_roe(current_price)

            if self.dca_phase:
                if self.position and roe <= -10:
                    pnl = self.close_position(current_price)
                    self.record_trade("DCA_STOP_LOSS", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = False
                    self.position_direction = "SHORT" if self.position_direction == "LONG" else "LONG"
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.reduced_tp = False
                    self.dca_stopped = True
                elif self.position and roe >= self.take_profit_roe:
                    pnl = self.close_position(current_price)
                    self.record_trade("DCA_TAKE_PROFIT", current_price, 0, roe, pnl, current_time)
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.reduced_tp = False
                    self.dca_stopped = False
                elif not self.dca_stopped and self.dca_count < self.max_dca_entries:
                    if margin_ratio < self.dca_margin_ratio or self.position is None:
                        if self.last_entry_time is None or (
                            current_time - self.last_entry_time
                        ).total_seconds() >= self.entry_interval:
                            quantity = self.calculate_entry_quantity(
                                price=current_price,
                                target_margin_ratio=self.dca_margin_ratio,
                                max_entries=self.max_dca_entries,
                                min_quantity=self.min_quantity,
                            )
                            self.open_position(self.position_direction, current_price, quantity, current_time)
                            self.dca_count += 1
                            self.last_entry_time = current_time
            else:
                current_tp = 3.0 if self.reduced_tp else self.take_profit_roe
                if self.position and roe >= current_tp:
                    pnl = self.close_position(current_price)
                    self.record_trade("MARTIN_TAKE_PROFIT", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = True
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.position_direction = "LONG"
                    self.reduced_tp = False
                    self.dca_stopped = False
                elif self.martingale_step < 8 and margin_ratio <= 20:
                    martingale_levels = [
                        (0, -5, 1),
                        (1, -5, 2),
                        (2, -5, 4),
                        (3, -10, 8),
                        (4, -20, 16),
                        (5, -50, 32),
                        (6, -70, 64),
                        (7, -100, 128),
                    ]
                    for step, roe_threshold, multiplier in martingale_levels:
                        if step == self.martingale_step and roe <= roe_threshold:
                            quantity = multiplier * self.min_quantity
                            self.open_position(self.position_direction, current_price, quantity, current_time)
                            self.martingale_step += 1
                            if step >= 5:
                                self.reduced_tp = True
                            break
                elif self.martingale_step >= 8 and self.position:
                    pnl = self.close_position(current_price)
                    self.record_trade("MARTIN_EXIT", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = True
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.position_direction = "LONG"
                    self.reduced_tp = False
                    self.dca_stopped = False

            self.record_equity(current_time, current_price)


class Logic4Simulator(TradingSimulator):
    """1000-split DCA with 8-step martingale."""

    def __init__(self, *args, coin_type: str = "SOL", **kwargs):
        super().__init__(*args, **kwargs)
        self.dca_phase = True
        self.dca_count = 0
        self.max_dca_entries = 1000
        self.entry_interval = 60
        self.last_entry_time: Optional[datetime] = None
        self.martingale_step = 0
        self.position_direction = "LONG"
        self.take_profit_roe = 5.0
        self.reduced_tp = False
        self.dca_stopped = False
        self.dca_margin_ratio = 5
        self.coin_type = coin_type
        self.min_quantity = 0.001 if coin_type == "BTC" else 0.01 if coin_type == "ETH" else 0.1

    def run(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            current_time = row["dt"]
            current_price = row["close"]

            margin_ratio = self.get_current_margin_ratio()
            roe = self.get_current_roe(current_price)

            if self.dca_phase:
                if self.position and roe <= -10:
                    pnl = self.close_position(current_price)
                    self.record_trade("DCA_STOP_LOSS", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = False
                    self.position_direction = "SHORT" if self.position_direction == "LONG" else "LONG"
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.reduced_tp = False
                    self.dca_stopped = True
                elif self.position and roe >= self.take_profit_roe:
                    pnl = self.close_position(current_price)
                    self.record_trade("DCA_TAKE_PROFIT", current_price, 0, roe, pnl, current_time)
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.dca_stopped = False
                elif not self.dca_stopped and self.dca_count < self.max_dca_entries:
                    if margin_ratio < self.dca_margin_ratio or self.position is None:
                        if self.last_entry_time is None or (
                            current_time - self.last_entry_time
                        ).total_seconds() >= self.entry_interval:
                            quantity = self.calculate_entry_quantity(
                                price=current_price,
                                target_margin_ratio=self.dca_margin_ratio,
                                max_entries=self.max_dca_entries,
                                min_quantity=self.min_quantity,
                            )
                            self.open_position(self.position_direction, current_price, quantity, current_time)
                            self.dca_count += 1
                            self.last_entry_time = current_time
            else:
                if self.position and roe <= -10:
                    pnl = self.close_position(current_price)
                    self.record_trade("MARTIN_STOP_LOSS", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = True
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.position_direction = "LONG"
                    self.reduced_tp = False
                    self.dca_stopped = False

                current_tp = 3.0 if self.reduced_tp else self.take_profit_roe
                if self.position and roe >= current_tp:
                    pnl = self.close_position(current_price)
                    self.record_trade("MARTIN_TAKE_PROFIT", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = True
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.position_direction = "LONG"
                    self.reduced_tp = False
                    self.dca_stopped = False
                elif self.martingale_step < 8 and margin_ratio <= 20:
                    martingale_levels = [
                        (0, -5, 1),
                        (1, -5, 2),
                        (2, -5, 4),
                        (3, -10, 8),
                        (4, -20, 16),
                        (5, -50, 32),
                        (6, -70, 64),
                        (7, -100, 128),
                    ]

                    for step, roe_threshold, multiplier in martingale_levels:
                        if step == self.martingale_step and roe <= roe_threshold:
                            quantity = multiplier * self.min_quantity
                            self.open_position(self.position_direction, current_price, quantity, current_time)
                            self.martingale_step += 1
                            if step >= 5:
                                self.reduced_tp = True
                            break

                if self.martingale_step >= 8 and self.position:
                    pnl = self.close_position(current_price)
                    self.record_trade("MARTIN_EXIT", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = True
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.position_direction = "LONG"
                    self.reduced_tp = False
                    self.dca_stopped = False

            self.record_equity(current_time, current_price)


class Logic5Simulator(TradingSimulator):
    """1000-split DCA with 8-step martingale (reduced stop rules)."""

    def __init__(self, *args, coin_type: str = "ETH", **kwargs):
        super().__init__(*args, **kwargs)
        self.dca_phase = True
        self.dca_count = 0
        self.max_dca_entries = 1000
        self.entry_interval = 60
        self.last_entry_time: Optional[datetime] = None
        self.martingale_step = 0
        self.position_direction = "LONG"
        self.take_profit_roe = 5.0
        self.reduced_tp = False
        self.dca_stopped = False
        self.coin_type = coin_type
        self.min_quantity = 0.001 if coin_type == "BTC" else 0.01 if coin_type == "ETH" else 0.1

    def run(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            current_time = row["dt"]
            current_price = row["close"]

            margin_ratio = self.get_current_margin_ratio()
            roe = self.get_current_roe(current_price)

            if self.dca_phase:
                if self.position and roe >= self.take_profit_roe:
                    pnl = self.close_position(current_price)
                    self.record_trade("DCA_TAKE_PROFIT", current_price, 0, roe, pnl, current_time)
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.dca_stopped = False
                elif not self.dca_stopped and self.dca_count < self.max_dca_entries:
                    if margin_ratio < 5 or self.position is None:
                        if self.last_entry_time is None or (
                            current_time - self.last_entry_time
                        ).total_seconds() >= self.entry_interval:
                            quantity = 0.5 / current_price
                            self.open_position(self.position_direction, current_price, quantity, current_time)
                            self.dca_count += 1
                            self.last_entry_time = current_time
            else:
                if self.martingale_step == 7 and self.position and roe <= -10:
                    pnl = self.close_position(current_price)
                    self.record_trade("MARTIN_STOP_LOSS", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = True
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.position_direction = "LONG"
                    self.reduced_tp = False
                    self.dca_stopped = False

                current_tp = 3.0 if self.reduced_tp else self.take_profit_roe
                if self.position and roe >= current_tp:
                    pnl = self.close_position(current_price)
                    self.record_trade("MARTIN_TAKE_PROFIT", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = True
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.position_direction = "LONG"
                    self.reduced_tp = False
                    self.dca_stopped = False
                elif self.martingale_step < 8 and margin_ratio <= 20:
                    martingale_levels = [
                        (0, -5, 1),
                        (1, -5, 2),
                        (2, -5, 4),
                        (3, -10, 8),
                        (4, -20, 16),
                        (5, -50, 32),
                        (6, -70, 64),
                        (7, -100, 128),
                    ]

                    for step, roe_threshold, multiplier in martingale_levels:
                        if step == self.martingale_step and roe <= roe_threshold:
                            quantity = multiplier * self.min_quantity
                            self.open_position(self.position_direction, current_price, quantity, current_time)
                            self.martingale_step += 1
                            if step >= 5:
                                self.reduced_tp = True
                            break

                if self.martingale_step >= 8 and self.position:
                    pnl = self.close_position(current_price)
                    self.record_trade("MARTIN_EXIT", current_price, 0, roe, pnl, current_time)
                    self.dca_phase = True
                    self.dca_count = 0
                    self.last_entry_time = None
                    self.martingale_step = 0
                    self.position_direction = "LONG"
                    self.reduced_tp = False
                    self.dca_stopped = False

            self.record_equity(current_time, current_price)


def run_all_simulations(
    df: pd.DataFrame, initial_capital: float = 10000, coin_type: str = "ETH"
) -> Dict[str, Dict[str, object]]:
    """Run Logic1~5 simulators."""
    results: Dict[str, Dict[str, object]] = {}

    sim1 = Logic1Simulator(initial_capital=initial_capital)
    sim1.run(df.copy())
    results["Logic1"] = {"simulator": sim1, "stats": sim1.get_statistics()}

    sim2 = Logic2Simulator(initial_capital=initial_capital, coin_type=coin_type)
    sim2.run(df.copy())
    results["Logic2"] = {"simulator": sim2, "stats": sim2.get_statistics()}

    sim3 = Logic3Simulator(initial_capital=initial_capital, coin_type=coin_type)
    sim3.run(df.copy())
    results["Logic3"] = {"simulator": sim3, "stats": sim3.get_statistics()}

    sim4 = Logic4Simulator(initial_capital=initial_capital, coin_type=coin_type)
    sim4.run(df.copy())
    results["Logic4"] = {"simulator": sim4, "stats": sim4.get_statistics()}

    sim5 = Logic5Simulator(initial_capital=initial_capital, coin_type=coin_type)
    sim5.run(df.copy())
    results["Logic5"] = {"simulator": sim5, "stats": sim5.get_statistics()}

    return results
