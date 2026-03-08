from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


root_dir = Path(__file__).resolve().parent.parent

TRADES_PATH = root_dir / "artifacts/reports/backtest_meta_model_v6_trades.csv"
REPORT_DIR = root_dir / "artifacts/reports"


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # probabilidad relevante según lado
    out["trade_prob"] = np.where(
        out["direction"] == "long",
        out["prob_long"],
        out["prob_short"],
    )

    out["trade_ev"] = np.where(
        out["direction"] == "long",
        out["ev_long_regime"],
        out["ev_short_regime"],
    )

    out["trade_setup_score"] = np.where(
        out["direction"] == "long",
        out["setup_score_long_regime"],
        out["setup_score_short_regime"],
    )

    # buckets
    out["prob_bucket"] = pd.cut(
        out["trade_prob"],
        bins=[0.0, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0],
        include_lowest=True
    )

    out["ev_bucket"] = pd.cut(
        out["trade_ev"],
        bins=[-1.0, 0.0, 0.001, 0.002, 0.004, 0.008, 1.0],
        include_lowest=True
    )

    out["score_bucket"] = pd.cut(
        out["trade_setup_score"],
        bins=[-1.0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0],
        include_lowest=True
    )

    out["size_bucket"] = pd.cut(
        out["size"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0 + 1e-9],
        include_lowest=True
    )

    out["win_flag"] = (out["net_ret"] > 0).astype(int)

    return out


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(group_col, dropna=False)
        .agg(
            n_trades=("net_ret", "count"),
            total_net_ret=("net_ret", "sum"),
            avg_trade_ret=("net_ret", "mean"),
            win_rate=("win_flag", "mean"),
            avg_size=("size", "mean"),
            avg_prob=("trade_prob", "mean"),
            avg_ev=("trade_ev", "mean"),
            avg_score=("trade_setup_score", "mean"),
        )
        .sort_values("total_net_ret", ascending=False)
        .reset_index()
    )

    return summary


def main():
    logger.info("=" * 80)
    logger.info("TRADE ATTRIBUTION ANALYSIS")
    logger.info("=" * 80)

    if not TRADES_PATH.exists():
        raise FileNotFoundError(f"No existe {TRADES_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(TRADES_PATH)
    if trades.empty:
        raise ValueError("El archivo de trades está vacío")

    # fechas
    if "timestamp_entry" in trades.columns:
        trades["timestamp_entry"] = pd.to_datetime(trades["timestamp_entry"])
    if "timestamp_exit" in trades.columns:
        trades["timestamp_exit"] = pd.to_datetime(trades["timestamp_exit"])

    trades = add_buckets(trades)

    # resumen global
    global_summary = {
        "n_trades": int(len(trades)),
        "total_net_ret": float(trades["net_ret"].sum()),
        "avg_trade_ret": float(trades["net_ret"].mean()),
        "win_rate": float((trades["net_ret"] > 0).mean()),
        "long_trades": int((trades["direction"] == "long").sum()),
        "short_trades": int((trades["direction"] == "short").sum()),
        "long_total_ret": float(trades.loc[trades["direction"] == "long", "net_ret"].sum()),
        "short_total_ret": float(trades.loc[trades["direction"] == "short", "net_ret"].sum()),
        "avg_prob_long_trades": float(trades.loc[trades["direction"] == "long", "prob_long"].mean()) if (trades["direction"] == "long").any() else 0.0,
        "avg_prob_short_trades": float(trades.loc[trades["direction"] == "short", "prob_short"].mean()) if (trades["direction"] == "short").any() else 0.0,
    }

    # attribution tables
    by_direction = summarize_group(trades, "direction")
    by_regime = summarize_group(trades, "regime_label")
    by_outcome = summarize_group(trades, "outcome")
    by_prob_bucket = summarize_group(trades, "prob_bucket")
    by_ev_bucket = summarize_group(trades, "ev_bucket")
    by_score_bucket = summarize_group(trades, "score_bucket")
    by_size_bucket = summarize_group(trades, "size_bucket")

    # direction x regime
    if not trades.empty:
        direction_regime = (
            trades.groupby(["direction", "regime_label"], dropna=False)
            .agg(
                n_trades=("net_ret", "count"),
                total_net_ret=("net_ret", "sum"),
                avg_trade_ret=("net_ret", "mean"),
                win_rate=("win_flag", "mean"),
            )
            .sort_values("total_net_ret", ascending=False)
            .reset_index()
        )
    else:
        direction_regime = pd.DataFrame()

    # guardar
    with open(REPORT_DIR / "trade_attribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    by_direction.to_csv(REPORT_DIR / "trade_attribution_by_direction.csv", index=False)
    by_regime.to_csv(REPORT_DIR / "trade_attribution_by_regime.csv", index=False)
    by_outcome.to_csv(REPORT_DIR / "trade_attribution_by_outcome.csv", index=False)
    by_prob_bucket.to_csv(REPORT_DIR / "trade_attribution_by_prob_bucket.csv", index=False)
    by_ev_bucket.to_csv(REPORT_DIR / "trade_attribution_by_ev_bucket.csv", index=False)
    by_score_bucket.to_csv(REPORT_DIR / "trade_attribution_by_score_bucket.csv", index=False)
    by_size_bucket.to_csv(REPORT_DIR / "trade_attribution_by_size_bucket.csv", index=False)
    direction_regime.to_csv(REPORT_DIR / "trade_attribution_direction_regime.csv", index=False)

    logger.info("=" * 80)
    logger.info("GLOBAL SUMMARY")
    logger.info("=" * 80)
    logger.info(json.dumps(global_summary, indent=2))

    logger.info("=" * 80)
    logger.info("TOP BY DIRECTION")
    logger.info("=" * 80)
    logger.info("\n" + by_direction.to_string(index=False))

    logger.info("=" * 80)
    logger.info("TOP BY REGIME")
    logger.info("=" * 80)
    logger.info("\n" + by_regime.to_string(index=False))

    logger.info("=" * 80)
    logger.info("TOP BY EV BUCKET")
    logger.info("=" * 80)
    logger.info("\n" + by_ev_bucket.to_string(index=False))

    logger.info("=" * 80)
    logger.info("TOP BY SCORE BUCKET")
    logger.info("=" * 80)
    logger.info("\n" + by_score_bucket.to_string(index=False))


if __name__ == "__main__":
    main()
