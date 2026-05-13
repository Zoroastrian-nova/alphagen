import json
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import pandas as pd

# --- Monkey-patch qlib to avoid 1min benchmark requirement ---
# qlib's _cal_benchmark always fetches data at 1min freq internally, but
# our data only has 5min resolution.  When benchmark=None we pass
# benchmark_config=None so the benchmark calculation is skipped entirely.
import qlib.backtest as _qlib_bt
from dataclasses_json import DataClassJsonMixin

_qlib_orig_get_strategy_executor = _qlib_bt.get_strategy_executor


def _qlib_patched_get_strategy_executor(*args, **kwargs):
    benchmark = kwargs.get("benchmark", None)
    _orig_create_account = _qlib_bt.create_account_instance

    # When benchmark is None, qlib still creates benchmark_config={} which
    # triggers _cal_benchmark -> get_higher_eq_freq_feature(freq="1min").
    # Force benchmark_config=None here to skip benchmark entirely.
    if benchmark is None:

        def _create_account_no_bench(
            *, start_time, end_time, benchmark=None, account=1e9, pos_type="Position"
        ):
            from qlib.backtest.account import Account

            if isinstance(account, (int, float)):
                init_cash = account
                position_dict = {}
            elif isinstance(account, dict):
                init_cash = account.pop("cash")
                position_dict = account
            else:
                raise ValueError("account must be in (int, float, dict)")
            return Account(
                init_cash=init_cash,
                position_dict=position_dict,
                pos_type=pos_type,
                benchmark_config=None,
            )

        _qlib_bt.create_account_instance = _create_account_no_bench
        try:
            return _qlib_orig_get_strategy_executor(*args, **kwargs)
        finally:
            _qlib_bt.create_account_instance = _orig_create_account
    else:
        return _qlib_orig_get_strategy_executor(*args, **kwargs)


_qlib_bt.get_strategy_executor = _qlib_patched_get_strategy_executor
# --- end monkey-patch ---

from qlib.backtest import backtest
from qlib.backtest import executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from qlib.contrib.strategy import TopkDropoutStrategy  # kept for compatibility

from alphagen.data.expression import *
from alphagen.data.parser import parse_expression
from alphagen.trade.strategy import TimeSeriesFuturesStrategy
from alphagen_generic.features import *
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_qlib.utils import load_alpha_pool_by_path


def _load_config(config_path: str = "symbol_config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(
    path: str, factory: Callable[[], _T], invalidate_cache: bool = False
) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


@dataclass
class BacktestResult(DataClassJsonMixin):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: Optional[str] = "SH000300",
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
        # TimeSeriesFuturesStrategy parameters
        long_threshold: float = 0.0,
        short_threshold: float = 0.0,
        risk_degree: float = 0.95,
        trade_unit: int = 100,
        only_tradable: bool = True,
    ):
        self._benchmark = benchmark
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost
        self._long_threshold = long_threshold
        self._short_threshold = short_threshold
        self._risk_degree = risk_degree
        self._trade_unit = trade_unit
        self._only_tradable = only_tradable

    def run(
        self,
        prediction: Union[pd.Series, pd.DataFrame],
        output_prefix: Optional[str] = None,
        account: int = 100_000_000,
        limit_threshold: float = 0.095,
    ) -> Tuple[pd.DataFrame, BacktestResult]:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        # Extract unique daily dates for executor (handles 5min -> day conversion)
        raw_dates = index.levels[0]
        if hasattr(raw_dates, "normalize"):
            dates = raw_dates.normalize().unique().sort_values()
        else:
            dates = raw_dates

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy = TimeSeriesFuturesStrategy(
                    signal=prediction,
                    long_threshold=self._long_threshold,
                    short_threshold=self._short_threshold,
                    risk_degree=self._risk_degree,
                    trade_unit=self._trade_unit,
                    only_tradable=self._only_tradable,
                )
                executor = exec.SimulatorExecutor(
                    time_per_step="day", generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=account,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": limit_threshold,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    },
                )[0]

        # qlib get_step_time always accesses calendar[idx+1] on last step.
        # Use -2 by default to avoid off-by-one; if that fails, try -3.
        last_idx = -2
        while True:
            try:
                portfolio_metric = backtest_impl(last_idx)
                break
            except IndexError:
                if last_idx <= -len(dates):
                    raise
                print(
                    f"Cannot backtest till last day, retrying with offset {last_idx - 1}"
                )
                last_idx -= 1

        report, _ = portfolio_metric["1day"]  # type: ignore
        result = self._analyze_report(report)
        try:
            graph = report_graph(report, show_notebook=False)[0]
        except (TypeError, ValueError):
            graph = None  # benchmark is disabled, graph can't be computed
        if output_prefix is not None:
            dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            if graph is not None:
                dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", result.to_json())
        return report, result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]  # type: ignore

        if "bench" in report.columns and not report["bench"].isna().all():
            excess = risk_analysis(report["return"] - report["bench"] - report["cost"])[
                "risk"
            ]
            return BacktestResult(
                sharpe=loc(returns, "information_ratio"),
                annual_return=loc(returns, "annualized_return"),
                max_drawdown=loc(returns, "max_drawdown"),
                information_ratio=loc(excess, "information_ratio"),
                annual_excess_return=loc(excess, "annualized_return"),
                excess_max_drawdown=loc(excess, "max_drawdown"),
            )
        else:
            return BacktestResult(
                sharpe=loc(returns, "information_ratio"),
                annual_return=loc(returns, "annualized_return"),
                max_drawdown=loc(returns, "max_drawdown"),
                information_ratio=0.0,
                annual_excess_return=0.0,
                excess_max_drawdown=0.0,
            )


if __name__ == "__main__":
    config = _load_config()
    bt_cfg = config["backtest"]
    paths_cfg = config["paths"]

    initialize_qlib(config["qlib_data_path"])

    # Optionally disable benchmark when data is not at 1min frequency
    benchmark = bt_cfg.get("benchmark", None)
    qlib_backtest = QlibBacktest(
        benchmark=benchmark,
        deal=bt_cfg["deal_price"],
        open_cost=bt_cfg["open_cost"],
        close_cost=bt_cfg["close_cost"],
        min_cost=bt_cfg["min_cost"],
        long_threshold=bt_cfg.get("long_threshold", 0.0),
        short_threshold=bt_cfg.get("short_threshold", 0.0),
        risk_degree=bt_cfg.get("risk_degree", 0.95),
        trade_unit=bt_cfg.get("trade_unit", 100),
        only_tradable=bt_cfg.get("only_tradable", True),
    )
    freq = config.get("freq", "day")
    data = StockData(
        instrument=bt_cfg["instrument"],
        start_time=bt_cfg["start_time"],
        end_time=bt_cfg["end_time"],
        freq=freq,
    )
    calc = QLibStockDataCalculator(data, None)

    def run_backtest(
        prefix: str, seed: int, exprs: List[Expression], weights: List[float]
    ):
        df = data.make_dataframe(calc.make_ensemble_alpha(exprs, weights))
        # qlib expects MultiIndex with named levels: datetime + instrument
        df.index.names = ["datetime", "instrument"]
        out_prefix = f"{paths_cfg['backtest_output']}/{prefix}/{seed}"
        qlib_backtest.run(
            df,
            output_prefix=out_prefix,
            account=bt_cfg["account"],
            limit_threshold=bt_cfg["limit_threshold"],
        )

    # --- GP backtest ---
    gp_output = Path(paths_cfg.get("gp_output", "out/gp"))
    gp_cfg = config.get("gp", {})
    gp_checkpoint = gp_cfg.get("symbolic_regressor", {}).get("generations", 40)
    if gp_output.exists():
        for p in gp_output.iterdir():
            if not p.is_dir():
                continue
            try:
                seed = int(p.name)
            except ValueError:
                continue
            state_path = p / f"{gp_checkpoint}.json"
            if not state_path.exists():
                continue
            with open(state_path) as f:
                report = json.load(f)
            res = report.get("res", {})
            inner = res.get("res", {}) if isinstance(res, dict) else {}
            state = inner.get("pool_state")
            if state is None:
                continue
            run_backtest(
                "gp",
                seed,
                [parse_expression(e) for e in state["exprs"]],
                state["weights"],
            )
            print(f"[GP] Backtested seed={seed}")

    # --- RL / LLM-enhanced results backtest ---
    results_dir = Path(paths_cfg.get("save", "out/results"))
    if results_dir.exists():
        for p in sorted(results_dir.iterdir()):
            if not p.is_dir():
                continue
            try:
                parts = p.name.split("_", 4)
                inst, size, seed_str, time_str, ver = parts
                size, seed = int(size), int(seed_str)
            except (ValueError, IndexError):
                continue
            pool_files = sorted(p.glob("*_steps_pool.json"))
            if not pool_files:
                continue
            latest_pool = pool_files[-1]
            exprs, weights = load_alpha_pool_by_path(str(latest_pool))
            run_backtest(ver, seed, exprs, weights)
            print(f"[Results] Backtested ver={ver} seed={seed}")

    # --- Pure LLM interaction backtest ---
    llm_inter = Path(
        paths_cfg.get("llm_tests_interaction", "out/llm-tests/interaction")
    )
    if llm_inter.exists():
        for p in llm_inter.iterdir():
            if not p.is_dir():
                continue
            report_path = p / "report.json"
            if not report_path.exists():
                continue
            with open(report_path) as f:
                report = json.load(f)
            if not isinstance(report, list) or len(report) == 0:
                continue
            state = report[-1].get("pool_state")
            if state is None:
                continue
            run_backtest(
                "llm",
                int(p.name.replace("v", "").split("_")[0])
                if p.name.startswith("v")
                else 0,
                [parse_expression(t[0]) for t in state],
                [t[1] for t in state],
            )
            print(f"[LLM] Backtested run={p.name}")
