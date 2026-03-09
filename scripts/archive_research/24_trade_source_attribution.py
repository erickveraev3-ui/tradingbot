from __future__ import annotations
import pandas as pd
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent

TRADES_PATH = root_dir / "artifacts/reports/secondary_layer_shadow_replay/simulated_trades.csv"


def analyze_by_source(df):

    print("\nSOURCE ATTRIBUTION")
    print("="*60)

    g = df.groupby("decision_source")

    summary = g.agg(
        trades=("net_ret", "count"),
        total_return=("net_ret", "sum"),
        avg_trade_return=("net_ret", "mean"),
        win_rate=("net_ret", lambda x: (x>0).mean())
    )

    print(summary)


def analyze_by_regime(df):

    print("\nREGIME ATTRIBUTION")
    print("="*60)

    g = df.groupby(["decision_source", "regime_label"])

    summary = g.agg(
        trades=("net_ret", "count"),
        total_return=("net_ret", "sum"),
        avg_trade_return=("net_ret", "mean"),
        win_rate=("net_ret", lambda x: (x>0).mean())
    )

    print(summary)


def analyze_ev_buckets(df):

    print("\nEV BUCKETS")
    print("="*60)

    bins = [-1, 0.002, 0.004, 0.006, 1]
    labels = ["low", "mid", "high", "elite"]

    if "ev_long_regime" in df.columns and "ev_short_regime" in df.columns:
        ev_used = df["ev_long_regime"].fillna(df["ev_short_regime"])
    elif "ev_used" in df.columns:
        ev_used = df["ev_used"]
    else:
        print("\nEV BUCKETS")
        print("="*60)
        print("No hay columnas EV en simulated_trades.csv")
        return

    df["ev_bucket"] = pd.cut(ev_used, bins=bins, labels=labels)

    g = df.groupby(["decision_source", "ev_bucket"])

    summary = g.agg(
        trades=("net_ret", "count"),
        avg_trade_return=("net_ret", "mean"),
        win_rate=("net_ret", lambda x: (x>0).mean())
    )

    print(summary)


def main():

    if not TRADES_PATH.exists():
        raise FileNotFoundError(TRADES_PATH)

    df = pd.read_csv(TRADES_PATH)

    analyze_by_source(df)

    analyze_by_regime(df)

    analyze_ev_buckets(df)


if __name__ == "__main__":
    main()
