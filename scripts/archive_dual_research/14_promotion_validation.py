from pathlib import Path
import pandas as pd
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = ROOT / "artifacts" / "reports" / "walkforward_dual_refine"

MODES = [
    "short_only_baseline",
    "dual_long_trendup_elite"
]

OUT_DIR = ROOT / "artifacts" / "reports" / "promotion_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Cargar métricas y trades
# ------------------------------------------------------------
def load_mode(mode):

    mode_dir = REPORT_DIR / mode

    trades = pd.read_csv(mode_dir / "trades.csv")
    folds = pd.read_csv(mode_dir / "folds.csv")

    with open(mode_dir / "metrics.json") as f:
        metrics = json.load(f)

    return trades, folds, metrics


# ------------------------------------------------------------
# Performance por mes
# ------------------------------------------------------------
def monthly_performance(trades):

    if len(trades) == 0:
        return pd.DataFrame()

    trades = trades.copy()

    trades["timestamp_entry"] = pd.to_datetime(trades["timestamp_entry"])

    trades["month"] = trades["timestamp_entry"].dt.to_period("M")

    monthly = trades.groupby("month").agg(
        n_trades=("net_ret","count"),
        total_return=("net_ret","sum"),
        avg_trade=("net_ret","mean"),
        win_rate=("net_ret", lambda x:(x>0).mean())
    )

    return monthly.reset_index()


# ------------------------------------------------------------
# Performance por régimen
# ------------------------------------------------------------
def regime_performance(trades):

    if len(trades)==0:
        return pd.DataFrame()

    g = trades.groupby("regime_label").agg(
        n_trades=("net_ret","count"),
        total_return=("net_ret","sum"),
        avg_trade=("net_ret","mean"),
        win_rate=("net_ret",lambda x:(x>0).mean())
    )

    return g.reset_index()


# ------------------------------------------------------------
# Actividad temporal
# ------------------------------------------------------------
def activity_stats(trades):

    if len(trades)<2:
        return {}

    ts = pd.to_datetime(trades["timestamp_entry"]).sort_values()

    diffs = ts.diff().dropna().dt.total_seconds()/3600

    return {
        "avg_hours_between_trades":float(diffs.mean()),
        "max_hours_between_trades":float(diffs.max())
    }


# ------------------------------------------------------------
# Ejecutar análisis
# ------------------------------------------------------------
summary = []

for mode in MODES:

    trades, folds, metrics = load_mode(mode)

    monthly = monthly_performance(trades)
    regime = regime_performance(trades)
    activity = activity_stats(trades)

    monthly.to_csv(OUT_DIR / f"{mode}_monthly.csv", index=False)
    regime.to_csv(OUT_DIR / f"{mode}_regime.csv", index=False)

    row = {
        "mode":mode,
        "total_return":metrics["total_return"],
        "sharpe":metrics["sharpe_ratio"],
        "drawdown":metrics["max_drawdown"],
        "n_trades":metrics["n_trades"],
        **activity
    }

    summary.append(row)

summary_df = pd.DataFrame(summary)

summary_df.to_csv(OUT_DIR / "promotion_summary.csv", index=False)

print("\n==============================")
print("PROMOTION SUMMARY")
print("==============================")
print(summary_df.to_string(index=False))

print("\nArchivos generados en:")
print(OUT_DIR)
