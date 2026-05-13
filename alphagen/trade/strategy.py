"""
Time-series futures strategy: each instrument trades independently based on
its own signal value. Signal > threshold → long, signal < -threshold → short.
"""

from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from qlib.backtest.decision import BaseTradeDecision, Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.utils.resam import resam_ts_data


# ── helper to coerce signal formats ──────────────────────────────
def _coerce_signal(
    signal: Union[
        Signal, Tuple[BaseModel, Dataset], List, Dict, str, pd.Series, pd.DataFrame
    ],
) -> Signal:
    # create_signal_from will cache pd.DataFrame/Series automatically
    return create_signal_from(signal)


class TimeSeriesFuturesStrategy(BaseStrategy):
    """
    Parameters
    ----------
    signal : Signal-compatible object
        Time-series signal per instrument.  Positive values indicate a
        long bias; negative values indicate a short bias.

    long_threshold : float, default 0.0
        Enter long when signal > long_threshold.

    short_threshold : float, default 0.0
        Enter short when signal < -short_threshold.

    exit_long_threshold : float, default None
        Exit long when signal drops below this value.  If None, uses
        ``long_threshold`` (symmetric).

    exit_short_threshold : float, default None
        Exit short when signal rises above this value.  If None, uses
        ``-short_threshold`` (symmetric).

    risk_degree : float, default 0.95
        Fraction of account value used for each position.

    trade_unit : int, default 100
        Rounding unit for trade amounts.

    only_tradable : bool, default True
        Skip instruments that are not tradable on a given day.
    """

    def __init__(
        self,
        *,
        signal=None,
        model=None,
        dataset=None,
        long_threshold: float = 0.0,
        short_threshold: float = 0.0,
        exit_long_threshold: Optional[float] = None,
        exit_short_threshold: Optional[float] = None,
        risk_degree: float = 0.95,
        trade_unit: int = 100,
        only_tradable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.signal: Signal = _coerce_signal(signal)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.exit_long_threshold = (
            exit_long_threshold if exit_long_threshold is not None else long_threshold
        )
        self.exit_short_threshold = (
            exit_short_threshold
            if exit_short_threshold is not None
            else -short_threshold
        )
        self.risk_degree = risk_degree
        self.trade_unit = trade_unit
        self.only_tradable = only_tradable

    # ── main entry point called by backtest loop ──────────────────
    def generate_trade_decision(
        self, execute_result=None
    ) -> Union[TradeDecisionWO, BaseTradeDecision]:
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(
            trade_step, shift=1
        )

        # 1. get today's signal ──────────────────────────────────
        pred_score = self.signal.get_signal(
            start_time=pred_start_time, end_time=pred_end_time
        )
        if pred_score is None:
            return TradeDecisionWO([], self)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]

        # 2. current portfolio ────────────────────────────────────
        current: Position = copy.deepcopy(self.trade_position)
        cash = current.get_cash()
        current_stocks: List[str] = current.get_stock_list()

        orders: List[Order] = []

        # 3. exit existing positions ──────────────────────────────
        for stock_id in current_stocks:
            amount = current.get_stock_amount(stock_id)
            if amount == 0:
                continue
            sig = pred_score.get(stock_id, np.nan)

            should_exit = False
            if amount > 0:  # currently long
                if sig < self.exit_long_threshold:
                    should_exit = True
            elif amount < 0:  # currently short
                if sig > self.exit_short_threshold:
                    should_exit = True

            if should_exit:
                orders.append(
                    Order(
                        stock_id=stock_id,
                        amount=abs(amount),
                        direction=Order.SELL if amount > 0 else Order.BUY,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    )
                )

        # 4. new entries ──────────────────────────────────────────
        for stock_id, sig in pred_score.items():
            if stock_id in current_stocks and current.get_stock_amount(stock_id) != 0:
                continue  # already in position

            if self.only_tradable and not self.trade_exchange.is_stock_tradable(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
            ):
                continue

            direction = None
            if sig > self.long_threshold:
                direction = Order.BUY
            elif sig < -self.short_threshold:
                direction = Order.SELL

            if direction is None:
                continue

            # scale position size by cash and close price
            price = self.trade_exchange.get_deal_price(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=direction,
            )
            if price is None or price <= 0:
                continue

            target_value = cash * self.risk_degree / max(1, 50)  # at most 50 positions
            amount = target_value / price
            amount = int(amount / self.trade_unit) * self.trade_unit
            if amount <= 0:
                continue

            orders.append(
                Order(
                    stock_id=stock_id,
                    amount=amount,
                    direction=direction,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
            )

        if not orders:
            return TradeDecisionWO([], self)

        # 5. validate orders via exchange ─────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid_orders = [o for o in orders if self.trade_exchange.check_order(o)]

        if not valid_orders:
            return TradeDecisionWO([], self)

        return TradeDecisionWO(valid_orders, self)
