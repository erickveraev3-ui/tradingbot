from __future__ import annotations

import sys
import json
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch


root_dir = Path(__file__).resolve().parent.parent
core_path = root_dir / "scripts" / "20_live_replay_engine.py"

spec = importlib.util.spec_from_file_location("live_replay_core", core_path)
live_replay_core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = live_replay_core
spec.loader.exec_module(live_replay_core)

DATA_PATH = root_dir / "data" / "processed" / "dataset_btc_triple_barrier_1h.csv"
MODEL_DIR = root_dir / "artifacts/models"
SCALER_DIR = root_dir / "artifacts/scalers"
OUT_DIR_DEFAULT = root_dir / "artifacts" / "reports" / "secondary_layer_sizer_v2"

DEVICE = live_replay_core.DEVICE
MetaModel = live_replay_core.MetaModel
CONFIG = dict(live_replay_core.CONFIG)

SECONDARY = {
    "long_threshold": 0.525,
    "long_prob_margin": 0.02,
    "long_min_ev": 0.0030,
    "long_min_setup": 1.45,

    "short_threshold_range": 0.55,
    "short_prob_margin_range": 0.03,
    "short_min_ev_range": 0.0035,
    "short_min_setup_range": 1.40,
}


def parse_args():
    p = argparse.ArgumentParser(description="Secondary layer replay con sizing v2.")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--start", type=str, default="")
    p.add_argument("--end", type=str, default="")
    p.add_argument("--fee", type=float, default=CONFIG["fee_rate"])
    p.add_argument("--slippage", type=float, default=CONFIG["slippage_bps"])
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def evaluate_secondary_long(row: pd.Series) -> Dict[str, Any]:
    regime = str(row["regime_label"])
    candidate = int(row.get("candidate_long", 0))
    prob = float(row["prob_long"])
    opp_prob = float(row["prob_short"])
    ev = float(row["ev_long_regime"])
    score = float(row["setup_score_long_regime"])

    detail = {
        "passed": False,
        "reason": "",
        "candidate": candidate,
        "prob": prob,
        "ev": ev,
        "score": score,
        "regime": regime,
    }

    if regime != "trend_up":
        detail["reason"] = "secondary_regime_block"
        return detail
    if candidate != 1:
        detail["reason"] = "secondary_no_candidate"
        return detail
    if prob < SECONDARY["long_threshold"]:
        detail["reason"] = "secondary_threshold_fail"
        return detail
    if (prob - opp_prob) < SECONDARY["long_prob_margin"]:
        detail["reason"] = "secondary_margin_fail"
        return detail
    if ev < SECONDARY["long_min_ev"]:
        detail["reason"] = "secondary_ev_fail"
        return detail
    if score < SECONDARY["long_min_setup"]:
        detail["reason"] = "secondary_score_fail"
        return detail

    detail["passed"] = True
    detail["reason"] = "secondary_passed_all"
    return detail


def evaluate_secondary_short(row: pd.Series) -> Dict[str, Any]:
    regime = str(row["regime_label"])
    candidate = int(row.get("candidate_short", 0))
    prob = float(row["prob_short"])
    opp_prob = float(row["prob_long"])
    ev = float(row["ev_short_regime"])
    score = float(row["setup_score_short_regime"])

    detail = {
        "passed": False,
        "reason": "",
        "candidate": candidate,
        "prob": prob,
        "ev": ev,
        "score": score,
        "regime": regime,
    }

    if regime != "range":
        detail["reason"] = "secondary_regime_block"
        return detail
    if candidate != 1:
        detail["reason"] = "secondary_no_candidate"
        return detail
    if prob < SECONDARY["short_threshold_range"]:
        detail["reason"] = "secondary_threshold_fail"
        return detail
    if (prob - opp_prob) < SECONDARY["short_prob_margin_range"]:
        detail["reason"] = "secondary_margin_fail"
        return detail
    if ev < SECONDARY["short_min_ev_range"]:
        detail["reason"] = "secondary_ev_fail"
        return detail
    if score < SECONDARY["short_min_setup_range"]:
        detail["reason"] = "secondary_score_fail"
        return detail

    detail["passed"] = True
    detail["reason"] = "secondary_passed_all"
    return detail


def regime_multiplier(decision_source: str, direction: str, regime: str) -> float:
    # Motor principal: algo más de convicción en su terreno natural
    if decision_source == "primary":
        if direction == "short" and regime == "trend_down":
            return 1.10
        if direction == "long" and regime == "trend_up":
            return 1.00
        if regime == "range":
            return 0.92
        return 0.95

    # Secondary: más prudente, pero útil
    if decision_source == "secondary":
        if direction == "long" and regime == "trend_up":
            return 0.65
        if direction == "short" and regime == "range":
            return 0.50
        return 0.40

    return 1.0


def source_cap(decision_source: str) -> float:
    if decision_source == "primary":
        return 0.60
    return 0.15


def source_floor(decision_source: str) -> float:
    if decision_source == "primary":
        return 0.10
    return 0.04


def compute_sizer_v2(
    decision_source: str,
    direction: str,
    regime: str,
    prob: float,
    threshold: float,
    setup_score: float,
    ev_used: float,
) -> float:
    base_size = live_replay_core.compute_side_size(
        direction=direction,
        prob=prob,
        threshold=threshold,
        setup_score=setup_score,
        ev_used=ev_used,
    )

    mult = regime_multiplier(decision_source, direction, regime)

    size = base_size * mult
    size = min(source_cap(decision_source), size)
    size = max(source_floor(decision_source), size)

    return float(size)


def main():
    args = parse_args()
    CONFIG["fee_rate"] = args.fee
    CONFIG["slippage_bps"] = args.slippage

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_model_path = MODEL_DIR / "meta_model_long_v3.pt"
    short_model_path = MODEL_DIR / "meta_model_short_v3.pt"
    long_scaler_path = SCALER_DIR / "meta_model_long_v3_scaler.json"
    short_scaler_path = SCALER_DIR / "meta_model_short_v3_scaler.json"

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["global_idx"] = np.arange(len(df), dtype=np.int64)

    if args.start:
        df = df[df["timestamp"] >= pd.to_datetime(args.start)].copy()
    if args.end:
        df = df[df["timestamp"] <= pd.to_datetime(args.end)].copy()

    if args.start or args.end:
        df_replay = df.reset_index(drop=True)
    else:
        df_replay = df.iloc[-min(args.bars, len(df)):].copy().reset_index(drop=True)

    long_scaler = live_replay_core.load_scaler_payload(long_scaler_path)
    short_scaler = live_replay_core.load_scaler_payload(short_scaler_path)

    X_long = live_replay_core.apply_robust_scaler(df_replay, long_scaler["feature_columns"], long_scaler)
    X_short = live_replay_core.apply_robust_scaler(df_replay, short_scaler["feature_columns"], short_scaler)

    long_model = MetaModel(input_dim=len(long_scaler["feature_columns"])).to(DEVICE)
    short_model = MetaModel(input_dim=len(short_scaler["feature_columns"])).to(DEVICE)
    long_model.load_state_dict(torch.load(long_model_path, map_location=DEVICE))
    short_model.load_state_dict(torch.load(short_model_path, map_location=DEVICE))
    long_model.eval()
    short_model.eval()

    with torch.no_grad():
        prob_long = torch.sigmoid(long_model(torch.tensor(X_long, dtype=torch.float32, device=DEVICE))).cpu().numpy().flatten()
        prob_short = torch.sigmoid(short_model(torch.tensor(X_short, dtype=torch.float32, device=DEVICE))).cpu().numpy().flatten()

    df_bt = df_replay.copy()
    df_bt["prob_long"] = prob_long
    df_bt["prob_short"] = prob_short

    atr = df_bt["atr"].replace(0, np.nan).ffill().bfill()
    close = df_bt["close"].replace(0, np.nan).ffill().bfill()
    df_bt["tp_long_pct"] = (CONFIG["tp_atr_mult_long"] * atr / close).clip(lower=0.0)
    df_bt["sl_long_pct"] = (CONFIG["sl_atr_mult_long"] * atr / close).clip(lower=0.0)
    df_bt["tp_short_pct"] = (CONFIG["tp_atr_mult_short"] * atr / close).clip(lower=0.0)
    df_bt["sl_short_pct"] = (CONFIG["sl_atr_mult_short"] * atr / close).clip(lower=0.0)

    df_bt = live_replay_core.compute_setup_scores(df_bt)
    roundtrip_cost_pct = 2 * (CONFIG["fee_rate"] + CONFIG["slippage_bps"] / 10000.0)
    df_bt = live_replay_core.compute_expected_values(df_bt, roundtrip_cost_pct=roundtrip_cost_pct)
    df_bt = live_replay_core.infer_market_regime(df_bt)
    df_bt = live_replay_core.apply_regime_adjustments(df_bt)
    df_bt = live_replay_core.select_regime_ev_trades(df_bt, side="long", top_percent=CONFIG["top_percent_long"], min_ev_regime=CONFIG["min_ev_regime"])
    df_bt = live_replay_core.select_regime_ev_trades(df_bt, side="short", top_percent=CONFIG["top_percent_short"], min_ev_regime=CONFIG["min_ev_regime"])

    capital = CONFIG["initial_capital"]
    equity_curve = []
    strategy_returns = []
    trade_log = []
    decision_trace = []

    counters = {"primary_passed": 0, "secondary_passed": 0, "no_trade": 0}
    i = 0
    cooldown = 0

    while i < len(df_bt) - 1:
        row = df_bt.iloc[i]
        regime = str(row["regime_label"])

        if cooldown > 0:
            decision_trace.append({
                "timestamp": str(row["timestamp"]),
                "decision": "NONE",
                "decision_source": "cooldown",
                "decision_reason": "cooldown_active",
                "size": 0.0,
            })
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            cooldown -= 1
            i += 1
            continue

        long_d = live_replay_core.evaluate_long_detail(row)
        short_d = live_replay_core.evaluate_short_detail(row)

        decision = "NONE"
        decision_source = "none"
        decision_reason = "no_side_passed"
        size = 0.0

        # Primary layer
        take_long = bool(long_d["passed"])
        take_short = bool(short_d["passed"])

        if take_long and take_short:
            rank_long = live_replay_core.conflict_rank("long", row)
            rank_short = live_replay_core.conflict_rank("short", row)
            if rank_long >= rank_short:
                take_short = False
                decision_reason = "long_won_conflict"
            else:
                take_long = False
                decision_reason = "short_won_conflict"

        if take_long:
            decision = "LONG"
            decision_source = "primary"
            decision_reason = "passed_all" if decision_reason == "no_side_passed" else decision_reason
            size = compute_sizer_v2(
                decision_source="primary",
                direction="long",
                regime=regime,
                prob=float(row["prob_long"]),
                threshold=CONFIG["long_threshold"],
                setup_score=float(row["setup_score_long_regime"]),
                ev_used=float(row["ev_long_regime"]),
            )
            counters["primary_passed"] += 1

        elif take_short:
            threshold = CONFIG["short_threshold_trend_down"] if regime == "trend_down" else CONFIG["short_threshold_range"]
            decision = "SHORT"
            decision_source = "primary"
            decision_reason = "passed_all" if decision_reason == "no_side_passed" else decision_reason
            size = compute_sizer_v2(
                decision_source="primary",
                direction="short",
                regime=regime,
                prob=float(row["prob_short"]),
                threshold=threshold,
                setup_score=float(row["setup_score_short_regime"]),
                ev_used=float(row["ev_short_regime"]),
            )
            counters["primary_passed"] += 1

        # Secondary layer if primary did nothing
        if decision == "NONE":
            sec_long = evaluate_secondary_long(row)
            sec_short = evaluate_secondary_short(row)

            take_sec_long = bool(sec_long["passed"])
            take_sec_short = bool(sec_short["passed"])

            if take_sec_long and take_sec_short:
                if float(row["ev_long_regime"]) >= float(row["ev_short_regime"]):
                    take_sec_short = False
                    decision_reason = "secondary_long_won_conflict"
                else:
                    take_sec_long = False
                    decision_reason = "secondary_short_won_conflict"

            if take_sec_long:
                decision = "LONG"
                decision_source = "secondary"
                decision_reason = sec_long["reason"]
                size = compute_sizer_v2(
                    decision_source="secondary",
                    direction="long",
                    regime=regime,
                    prob=float(row["prob_long"]),
                    threshold=SECONDARY["long_threshold"],
                    setup_score=float(row["setup_score_long_regime"]),
                    ev_used=float(row["ev_long_regime"]),
                )
                counters["secondary_passed"] += 1

            elif take_sec_short:
                decision = "SHORT"
                decision_source = "secondary"
                decision_reason = sec_short["reason"]
                size = compute_sizer_v2(
                    decision_source="secondary",
                    direction="short",
                    regime=regime,
                    prob=float(row["prob_short"]),
                    threshold=SECONDARY["short_threshold_range"],
                    setup_score=float(row["setup_score_short_regime"]),
                    ev_used=float(row["ev_short_regime"]),
                )
                counters["secondary_passed"] += 1

        if decision == "NONE":
            counters["no_trade"] += 1

        decision_trace.append({
            "timestamp": str(row["timestamp"]),
            "close": float(row["close"]),
            "regime_label": regime,
            "decision": decision,
            "decision_source": decision_source,
            "decision_reason": decision_reason,
            "size": size,
            "prob_long": float(row["prob_long"]),
            "prob_short": float(row["prob_short"]),
            "ev_long_regime": float(row["ev_long_regime"]),
            "ev_short_regime": float(row["ev_short_regime"]),
            "setup_score_long_regime": float(row["setup_score_long_regime"]),
            "setup_score_short_regime": float(row["setup_score_short_regime"]),
            "candidate_long": int(row.get("candidate_long", 0)),
            "candidate_short": int(row.get("candidate_short", 0)),
        })

        if decision == "NONE":
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            i += 1
            continue

        direction = "long" if decision == "LONG" else "short"
        gross_ret, exit_idx, outcome = live_replay_core.simulate_trade(df_bt, i, direction)

        fee_cost = live_replay_core.cost_for_transition(0.0, size, CONFIG["fee_rate"], CONFIG["slippage_bps"])
        fee_cost += live_replay_core.cost_for_transition(size, 0.0, CONFIG["fee_rate"], CONFIG["slippage_bps"])
        net_ret = size * gross_ret - fee_cost
        capital *= (1.0 + net_ret)

        trade_log.append({
            "entry_idx": i,
            "exit_idx": exit_idx,
            "timestamp_entry": str(df_bt.iloc[i]["timestamp"]),
            "timestamp_exit": str(df_bt.iloc[min(exit_idx, len(df_bt)-1)]["timestamp"]),
            "direction": direction,
            "decision_source": decision_source,
            "decision_reason": decision_reason,
            "regime_label": regime,
            "size": size,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "outcome": outcome,
            "prob_long": float(row["prob_long"]),
            "prob_short": float(row["prob_short"]),
            "ev_long_regime": float(row["ev_long_regime"]),
            "ev_short_regime": float(row["ev_short_regime"]),
            "setup_score_long_regime": float(row["setup_score_long_regime"]),
            "setup_score_short_regime": float(row["setup_score_short_regime"]),
        })

        equity_curve.append(capital)
        strategy_returns.append(net_ret)
        i = max(i + 1, exit_idx + 1)
        cooldown = CONFIG["cooldown_bars"]

    decision_df = pd.DataFrame(decision_trace)
    trades_df = pd.DataFrame(trade_log)
    equity_curve = np.array(equity_curve, dtype=np.float64)
    strategy_returns = np.array(strategy_returns, dtype=np.float64)

    summary = {
        "bars_replayed": int(len(df_bt)),
        "start_timestamp": str(df_bt["timestamp"].iloc[0]) if len(df_bt) else "",
        "end_timestamp": str(df_bt["timestamp"].iloc[-1]) if len(df_bt) else "",
        "initial_capital": CONFIG["initial_capital"],
        "final_capital": float(capital),
        "total_return": float(capital / CONFIG["initial_capital"] - 1.0),
        "max_drawdown": live_replay_core.max_drawdown(equity_curve),
        "sharpe_ratio": live_replay_core.sharpe_ratio(strategy_returns),
        "sortino_ratio": live_replay_core.sortino_ratio(strategy_returns),
        "n_trades": int(len(trades_df)),
        "avg_trade_return": float(trades_df["net_ret"].mean()) if len(trades_df) else 0.0,
        "win_rate_trade": float((trades_df["net_ret"] > 0).mean()) if len(trades_df) else 0.0,
        "long_trades": int((trades_df["direction"] == "long").sum()) if len(trades_df) else 0,
        "short_trades": int((trades_df["direction"] == "short").sum()) if len(trades_df) else 0,
        "primary_trades": int((trades_df["decision_source"] == "primary").sum()) if len(trades_df) else 0,
        "secondary_trades": int((trades_df["decision_source"] == "secondary").sum()) if len(trades_df) else 0,
        "avg_primary_size": float(trades_df.loc[trades_df["decision_source"] == "primary", "size"].mean()) if len(trades_df) else 0.0,
        "avg_secondary_size": float(trades_df.loc[trades_df["decision_source"] == "secondary", "size"].mean()) if len(trades_df) else 0.0,
        "counters": counters,
    }

    decision_df.to_csv(out_dir / "decision_trace.csv", index=False)
    trades_df.to_csv(out_dir / "simulated_trades.csv", index=False)
    pd.DataFrame({
        "step": np.arange(len(equity_curve)),
        "equity": equity_curve,
        "strategy_return": strategy_returns,
    }).to_csv(out_dir / "equity_curve.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSIZER V2 SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
