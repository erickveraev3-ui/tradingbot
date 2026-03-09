from __future__ import annotations

import json
import subprocess
import argparse
from pathlib import Path
from datetime import timedelta

import pandas as pd


root_dir = Path(__file__).resolve().parent.parent

REPLAY_SCRIPT = root_dir / "scripts" / "22_secondary_layer_shadow_replay.py"
DATA_PATH = root_dir / "data" / "processed" / "dataset_btc_triple_barrier_1h.csv"

OUT_DIR = root_dir / "artifacts" / "reports" / "secondary_layer_rolling"


def parse_args():
    p = argparse.ArgumentParser(description="Rolling replay scanner para estrategia con secondary layer.")
    p.add_argument("--window_days", type=int, default=30)
    p.add_argument("--step_days", type=int, default=7)
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default="")
    p.add_argument("--fee", type=float, default=0.0008)
    p.add_argument("--slippage", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else df["timestamp"].max()

    cur_start = start
    rows = []
    idx = 0

    while cur_start < end:

        cur_end = cur_start + pd.Timedelta(days=args.window_days)

        if cur_end > end:
            break

        run_dir = OUT_DIR / f"run_{idx:03d}_{cur_start.date()}_{cur_end.date()}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(REPLAY_SCRIPT),
            "--start", str(cur_start.date()),
            "--end", str(cur_end.date()),
            "--fee", str(args.fee),
            "--slippage", str(args.slippage),
            "--out_dir", str(run_dir),
            "--quiet",
        ]

        subprocess.run(cmd, check=True)

        summary_path = run_dir / "summary.json"

        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)

        rows.append({
            "window_start": str(cur_start.date()),
            "window_end": str(cur_end.date()),

            "bars_replayed": s["bars_replayed"],

            "total_return": s["total_return"],
            "max_drawdown": s["max_drawdown"],
            "sharpe_ratio": s["sharpe_ratio"],
            "sortino_ratio": s["sortino_ratio"],

            "n_trades": s["n_trades"],
            "long_trades": s["long_trades"],
            "short_trades": s["short_trades"],

            "primary_trades": s["primary_trades"],
            "secondary_trades": s["secondary_trades"],

            "avg_trade_return": s["avg_trade_return"],
            "win_rate_trade": s["win_rate_trade"],
        })

        cur_start = cur_start + pd.Timedelta(days=args.step_days)
        idx += 1

    out_df = pd.DataFrame(rows)

    out_df.to_csv(OUT_DIR / "secondary_rolling_summary.csv", index=False)

    print("\nSECONDARY LAYER ROLLING SUMMARY")
    print(out_df.to_string(index=False))

    if len(out_df) > 0:
        print("\nAGGREGATE METRICS")
        print(
            out_df[
                [
                    "total_return",
                    "max_drawdown",
                    "sharpe_ratio",
                    "n_trades",
                    "primary_trades",
                    "secondary_trades",
                ]
            ].describe()
        )


if __name__ == "__main__":
    main()
