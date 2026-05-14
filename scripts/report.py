"""
Factor evaluation & backtest visualization report.

Usage:
    python scripts/report.py
    python scripts/report.py --config symbol_config.json
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ── config helpers ────────────────────────────────────────────────
def _load_config(config_path: str = "symbol_config.json") -> dict:
    with open(config_path) as f:
        return json.load(f)


# ── factor analysis ───────────────────────────────────────────────
def compute_factor_analysis(
    pool_dir: Path,
    config: dict,
    out_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    For the latest checkpoint in *pool_dir*, parse each alpha expression,
    compute cross-sectional IC / Rank IC against next-day returns,
    and produce a factor correlation matrix.
    """
    from alphagen.data.expression import Expression
    from alphagen.data.parser import parse_expression
    from alphagen_qlib.calculator import QLibStockDataCalculator
    from alphagen_qlib.stock_data import StockData, initialize_qlib

    initialize_qlib(config["qlib_data_path"])

    bt_cfg = config["backtest"]
    data = StockData(
        instrument=bt_cfg["instrument"],
        start_time=bt_cfg["start_time"],
        end_time=bt_cfg["end_time"],
        freq=config.get("freq", "day"),
        max_backtrack_days=100,
        max_future_days=30,
    )
    calc = QLibStockDataCalculator(data, None)
    n_days, n_stocks = data.n_days, data.n_stocks

    # find latest pool checkpoint
    pool_files = sorted(pool_dir.glob("*_steps_pool.json"))
    if not pool_files:
        print(f"[factor] No pool checkpoint found in {pool_dir}")
        return None
    latest = pool_files[-1]
    print(f"[factor] Loading pool from {latest}")
    with open(latest) as f:
        pool = json.load(f)

    exprs: List[Expression] = [parse_expression(e) for e in pool["exprs"]]
    weights = pool.get("weights", [1.0 / len(exprs)] * len(exprs))
    n_factors = len(exprs)

    # compute forward return (next-day close / close - 1) per stock per day
    # align with evaluate_alpha output window
    close_data = data.data[:, 0, :].cpu().numpy()  # shape (T_full, n_stocks)
    bt = data.max_backtrack_days
    fut = data.max_future_days
    # forward return on the same n_days window as evaluate_alpha
    fwd_ret = (close_data[bt + 1 : bt + n_days + 1, :]
               / close_data[bt : bt + n_days, :] - 1)
    fwd_ret[np.isinf(fwd_ret)] = np.nan

    # --- individual factor values ---
    factor_vals: Dict[str, np.ndarray] = {}
    print(f"[factor] Computing {n_factors} factors...")
    for i, expr in enumerate(exprs):
        val = calc.evaluate_alpha(expr)  # (T, n_stocks), normalized
        val = val.cpu().numpy()
        name = f"alpha_{i:02d}"
        factor_vals[name] = val
        if i < 3:
            print(f"  {name}: {str(expr)[:80]}...")

    # --- IC table ---
    rows = []
    for name, val in factor_vals.items():
        # day-wise cross-sectional IC
        daily_ic = []
        daily_rank_ic = []
        for d in range(n_days - 1):
            fv = val[d, :]
            fr = fwd_ret[d, :]
            mask = ~np.isnan(fv) & ~np.isnan(fr)
            if mask.sum() < 5:
                continue
            ic, _ = pearsonr(fv[mask], fr[mask])
            rank_ic, _ = spearmanr(fv[mask], fr[mask])
            daily_ic.append(ic)
            daily_rank_ic.append(rank_ic)
        daily_ic = np.array(daily_ic)
        daily_rank_ic = np.array(daily_rank_ic)
        rows.append(
            {
                "factor": name,
                "weight": weights[i],
                "IC_mean": np.nanmean(daily_ic),
                "IC_std": np.nanstd(daily_ic),
                "IC_ir": np.nanmean(daily_ic) / (np.nanstd(daily_ic) + 1e-12),
                "RankIC_mean": np.nanmean(daily_rank_ic),
                "RankIC_std": np.nanstd(daily_rank_ic),
                "RankIC_ir": np.nanmean(daily_rank_ic)
                / (np.nanstd(daily_rank_ic) + 1e-12),
            }
        )

    ic_df = pd.DataFrame(rows).set_index("factor")
    ic_df.to_csv(out_dir / "factor_ic.csv")
    print(f"[factor] Saved factor_ic.csv ({len(ic_df)} rows)")

    # --- correlation matrix ---
    # flatten each factor across days × stocks (use last 30 days to reduce noise)
    recent = max(0, n_days - 30)
    samples = []
    labels = []
    for name, val in factor_vals.items():
        flat = val[recent:, :].reshape(-1)
        mask = ~np.isnan(flat)
        if mask.sum() < 10:
            continue
        samples.append(flat[mask])
        labels.append(name)
    if len(samples) >= 2:
        corr = np.corrcoef(samples)
        # ensure symmetric; truncate to matching labels
        n = len(samples)
        corr_df = pd.DataFrame(corr[:n, :n], index=labels, columns=labels)
        corr_df.to_csv(out_dir / "factor_correlation.csv")

        fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(6, n * 0.4)))
        im = ax.matshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Factor Correlation Matrix")
        fig.savefig(out_dir / "factor_correlation.png")
        plt.close(fig)
        print(f"[factor] Saved factor_correlation.png")

    # --- weights bar chart ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, weights[:n_factors], color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Factor Weights")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Weight")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    fig.savefig(out_dir / "factor_weights.png")
    plt.close(fig)

    return ic_df


# ── backtest visualization ────────────────────────────────────────
def plot_backtest(report_dir: Path, out_dir: Path, label: str) -> None:
    """Load report.pkl and save equity curve + drawdown chart."""
    reports = sorted(report_dir.glob("*-report.pkl"))
    if not reports:
        print(f"[viz] No report found in {report_dir}")
        return
    report_path = reports[-1]  # use the latest one

    with open(report_path, "rb") as f:
        report: pd.DataFrame = pickle.load(f)

    # cumulative return
    cum_ret = (1 + report["return"]).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # top: equity curve
    axes[0].plot(cum_ret.index, cum_ret.values, color="steelblue", linewidth=0.8)
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].set_title(f"Backtest Equity Curve — {label}")
    axes[0].grid(True, alpha=0.3)

    # middle: drawdown
    roll_max = cum_ret.expanding().max()
    dd = cum_ret / roll_max - 1
    axes[1].fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)

    # bottom: daily return bar
    axes[2].bar(
        report.index, report["return"].values, width=1.0, color="steelblue", alpha=0.7
    )
    axes[2].set_ylabel("Daily Return")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    for ax in axes:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)

    fig.savefig(out_dir / f"equity_{label}.png")
    plt.close(fig)
    print(f"[viz] Saved equity_{label}.png")

    # summary table
    result_path = report_dir / "0-result.json"
    if result_path.exists():
        with open(result_path) as f:
            result = json.load(f)
        summary = pd.DataFrame([result])
        summary.to_csv(out_dir / f"summary_{label}.csv", index=False)
        print(f"[viz] Saved summary_{label}.csv")


# ── main ──────────────────────────────────────────────────────────
def main(
    config_path: str = "symbol_config.json",
    results_dir: str = "out/results",
    backtest_dir: str = "out/backtests",
    output_dir: str = "out/report",
):
    """Run factor analysis and backtest visualization.

    Parameters
    ----------
    config_path : str
        Path to symbol_config.json.
    results_dir : str
        Directory containing RL/LLM result subdirectories (pool checkpoints).
    backtest_dir : str
        Directory containing backtest output (report.pkl, result.json).
    output_dir : str
        Where to save the report files.
    """
    config = _load_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── factor analysis ────────────────────────────────────────
    pool_root = Path(results_dir)
    if pool_root.exists():
        for sub in sorted(pool_root.iterdir()):
            if not sub.is_dir():
                continue
            if not list(sub.glob("*_steps_pool.json")):
                continue
            tag = sub.name.rsplit("_", 1)[-1]  # rl / llm 等
            print(f"\n{'=' * 60}\nFactor analysis for {sub.name}\n{'=' * 60}")
            compute_factor_analysis(sub, config, out)

    # ── backtest visualization ─────────────────────────────────
    bt_root = Path(backtest_dir)
    if bt_root.exists():
        for sub in sorted(bt_root.iterdir()):
            if not sub.is_dir():
                continue
            print(f"\n{'=' * 60}\nBacktest plot for {sub.name}\n{'=' * 60}")
            plot_backtest(sub, out, sub.name)

    print(f"\nAll reports saved to {out.resolve()}/")


if __name__ == "__main__":
    fire.Fire(main)
