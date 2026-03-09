from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from src.strategy.setup_ranking import compute_setup_scores
from src.strategy.position_sizer import dynamic_position_size
from src.strategy.expected_value_engine import compute_expected_values
from src.strategy.regime_engine import infer_market_regime, apply_regime_adjustments


# =============================================================================
# PATHS
# =============================================================================

DATA_PATH = root_dir / "data/processed/dataset_btc_triple_barrier_1h.csv"
MODEL_DIR = root_dir / "artifacts/models"
SCALER_DIR = root_dir / "artifacts/scalers"
OUT_DIR_DEFAULT = root_dir / "artifacts/reports/paper_trace_simulator"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# CONFIG - CHAMPION CONGELADO
# =============================================================================

CONFIG = {
    "initial_capital": 10000.0,
    "fee_rate": 0.0008,          # escenario base realista
    "slippage_bps": 5.0,

    "cooldown_bars": 1,

    # sizing
    "base_size": 0.10,
    "max_size": 0.60,
    "prob_scale": 2.0,
    "score_scale": 0.20,

    # mismas barreras del dataset
    "tp_atr_mult_long": 1.8,
    "sl_atr_mult_long": 1.2,
    "tp_atr_mult_short": 1.8,
    "sl_atr_mult_short": 1.2,

    # selección previa por EV/régimen
    "min_ev_regime": 0.002,
    "top_percent_long": 0.35,
    "top_percent_short": 0.40,

    # champion long
    "long_threshold": 0.555,
    "long_prob_margin": 0.05,
    "long_min_ev": 0.0055,
    "long_min_setup": 1.90,

    # champion short
    "short_threshold_trend_down": 0.525,
    "short_prob_margin_trend_down": 0.03,
    "short_min_ev_trend_down": 0.0038,
    "short_min_setup_trend_down": 1.35,

    "short_threshold_range": 0.54,
    "short_prob_margin_range": 0.04,
    "short_min_ev_range": 0.0048,
    "short_min_setup_range": 1.55,
}


# =============================================================================
# MODEL
# =============================================================================

class MetaModel(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# UTILS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Paper trace simulator del champion actual.")
    parser.add_argument("--bars", type=int, default=500, help="Número de barras finales a simular.")
    parser.add_argument("--fee", type=float, default=CONFIG["fee_rate"], help="Fee por transición.")
    parser.add_argument("--slippage", type=float, default=CONFIG["slippage_bps"], help="Slippage en bps.")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT), help="Carpeta de salida.")
    parser.add_argument("--quiet", action="store_true", help="Reduce prints en terminal.")
    return parser.parse_args()


def load_scaler_payload(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_robust_scaler(df: pd.DataFrame, feature_cols: list[str], scaler_payload: dict) -> np.ndarray:
    x = df[feature_cols].values.astype(np.float32)
    center = np.array(scaler_payload["center"], dtype=np.float32)
    scale = np.array(scaler_payload["scale"], dtype=np.float32) + 1e-8
    x = (x - center) / scale
    x = np.clip(x, -8, 8)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def cost_for_transition(pos_prev: float, pos_new: float, fee_rate: float, slippage_bps: float) -> float:
    turnover = abs(pos_new - pos_prev)
    if turnover == 0:
        return 0.0
    return turnover * fee_rate + turnover * (slippage_bps / 10000.0)


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / np.maximum(running_max, 1e-12)
    return float(dd.min())


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 24 * 365) -> float:
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    if sigma < 1e-12:
        return 0.0
    return mu / sigma * np.sqrt(periods_per_year)


def sortino_ratio(returns: np.ndarray, periods_per_year: int = 24 * 365) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma_down = float(np.std(downside))
    if sigma_down < 1e-12:
        return 0.0
    return mu / sigma_down * np.sqrt(periods_per_year)


def select_regime_ev_trades(df: pd.DataFrame, side: str, top_percent: float, min_ev_regime: float) -> pd.DataFrame:
    out = df.copy()

    if side == "long":
        mask = (out["candidate_long"] == 1) & (out["ev_long_regime"] > min_ev_regime)
        score_col = "ev_long_regime"
        flag_col = "selected_long_regime"
    else:
        mask = (out["candidate_short"] == 1) & (out["ev_short_regime"] > min_ev_regime)
        score_col = "ev_short_regime"
        flag_col = "selected_short_regime"

    out[flag_col] = 0

    if mask.any():
        thr = out.loc[mask, score_col].quantile(max(0.0, 1.0 - top_percent))
        out.loc[mask & (out[score_col] >= thr), flag_col] = 1

    return out


def compute_side_size(direction: str, prob: float, threshold: float, setup_score: float, ev_used: float) -> float:
    score_input = setup_score + max(0.0, 50.0 * ev_used)
    size = dynamic_position_size(
        prob=prob,
        threshold=threshold,
        setup_score=score_input,
        base_size=CONFIG["base_size"],
        max_size=CONFIG["max_size"],
        prob_scale=CONFIG["prob_scale"],
        score_scale=CONFIG["score_scale"],
    )
    return float(min(CONFIG["max_size"], max(CONFIG["base_size"], size)))


def conflict_rank(direction: str, row: pd.Series) -> float:
    if direction == "short":
        p_short = float(row["prob_short"])
        ev_short = float(row["ev_short_regime"])
        score_short = float(row["setup_score_short_regime"])
        return (
            1.00 * ev_short +
            0.18 * max(0.0, p_short - 0.53) +
            0.03 * score_short
        )

    p_long = float(row["prob_long"])
    p_short = float(row["prob_short"])
    ev_long = float(row["ev_long_regime"])
    score_long = float(row["setup_score_long_regime"])

    rank = (
        1.00 * ev_long +
        0.14 * max(0.0, p_long - CONFIG["long_threshold"]) +
        0.025 * score_long
    )

    if p_short >= 0.53:
        rank -= 0.0015
    if float(row["ev_short_regime"]) >= 0.0045:
        rank -= 0.0015
    if (p_long - p_short) < 0.06:
        rank -= 0.0010

    return rank


def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str) -> Tuple[float, int, str]:
    entry_global_idx = int(df.iloc[entry_idx]["global_idx"])

    if direction == "long":
        label = int(df.iloc[entry_idx]["tb_long_label"])
        ret = float(df.iloc[entry_idx]["tb_long_return"])
        hit_bar_global = int(df.iloc[entry_idx]["tb_long_hit_bar"])
    else:
        label = int(df.iloc[entry_idx]["tb_short_label"])
        ret = float(df.iloc[entry_idx]["tb_short_return"])
        hit_bar_global = int(df.iloc[entry_idx]["tb_short_hit_bar"])

    delta = hit_bar_global - entry_global_idx
    if delta < 1:
        delta = 1

    exit_idx_local = min(entry_idx + delta, len(df) - 1)

    if label == 1:
        outcome = "tp"
    elif label == -1:
        outcome = "sl"
    else:
        outcome = "expiry"

    return ret, exit_idx_local, outcome


# =============================================================================
# DECISION ENGINE
# =============================================================================

def evaluate_long_detail(row: pd.Series) -> Dict[str, Any]:
    regime = str(row["regime_label"])
    candidate = int(row.get("candidate_long", 0))
    selected_regime = int(row.get("selected_long_regime", 0))

    prob = float(row["prob_long"])
    opp_prob = float(row["prob_short"])
    ev = float(row["ev_long_regime"])
    score = float(row["setup_score_long_regime"])

    detail = {
        "side": "long",
        "candidate": candidate,
        "selected_regime": selected_regime,
        "prob": prob,
        "opp_prob": opp_prob,
        "ev": ev,
        "score": score,
        "threshold": CONFIG["long_threshold"],
        "prob_margin": CONFIG["long_prob_margin"],
        "min_ev": CONFIG["long_min_ev"],
        "min_setup": CONFIG["long_min_setup"],
        "regime": regime,
        "passed": False,
        "reason": "",
        "threshold_ok": 0,
        "margin_ok": 0,
        "ev_ok": 0,
        "score_ok": 0,
    }

    if regime != "trend_up":
        detail["reason"] = "regime_block"
        return detail

    if candidate != 1:
        detail["reason"] = "no_candidate"
        return detail

    if selected_regime != 1:
        detail["reason"] = "not_selected_regime"
        return detail

    detail["threshold_ok"] = int(prob >= CONFIG["long_threshold"])
    detail["margin_ok"] = int((prob - opp_prob) >= CONFIG["long_prob_margin"])
    detail["ev_ok"] = int(ev >= CONFIG["long_min_ev"])
    detail["score_ok"] = int(score >= CONFIG["long_min_setup"])

    if not detail["threshold_ok"]:
        detail["reason"] = "threshold_fail"
        return detail
    if not detail["margin_ok"]:
        detail["reason"] = "margin_fail"
        return detail
    if not detail["ev_ok"]:
        detail["reason"] = "ev_fail"
        return detail
    if not detail["score_ok"]:
        detail["reason"] = "score_fail"
        return detail

    detail["passed"] = True
    detail["reason"] = "passed_all"
    return detail


def evaluate_short_detail(row: pd.Series) -> Dict[str, Any]:
    regime = str(row["regime_label"])
    candidate = int(row.get("candidate_short", 0))
    selected_regime = int(row.get("selected_short_regime", 0))

    prob = float(row["prob_short"])
    opp_prob = float(row["prob_long"])
    ev = float(row["ev_short_regime"])
    score = float(row["setup_score_short_regime"])

    if regime == "trend_down":
        threshold = CONFIG["short_threshold_trend_down"]
        prob_margin = CONFIG["short_prob_margin_trend_down"]
        min_ev = CONFIG["short_min_ev_trend_down"]
        min_setup = CONFIG["short_min_setup_trend_down"]
    elif regime == "range":
        threshold = CONFIG["short_threshold_range"]
        prob_margin = CONFIG["short_prob_margin_range"]
        min_ev = CONFIG["short_min_ev_range"]
        min_setup = CONFIG["short_min_setup_range"]
    else:
        threshold = None
        prob_margin = None
        min_ev = None
        min_setup = None

    detail = {
        "side": "short",
        "candidate": candidate,
        "selected_regime": selected_regime,
        "prob": prob,
        "opp_prob": opp_prob,
        "ev": ev,
        "score": score,
        "threshold": threshold,
        "prob_margin": prob_margin,
        "min_ev": min_ev,
        "min_setup": min_setup,
        "regime": regime,
        "passed": False,
        "reason": "",
        "threshold_ok": 0,
        "margin_ok": 0,
        "ev_ok": 0,
        "score_ok": 0,
    }

    if regime not in ("trend_down", "range"):
        detail["reason"] = "regime_block"
        return detail

    if candidate != 1:
        detail["reason"] = "no_candidate"
        return detail

    if selected_regime != 1:
        detail["reason"] = "not_selected_regime"
        return detail

    detail["threshold_ok"] = int(prob >= threshold)
    detail["margin_ok"] = int((prob - opp_prob) >= prob_margin)
    detail["ev_ok"] = int(ev >= min_ev)
    detail["score_ok"] = int(score >= min_setup)

    if not detail["threshold_ok"]:
        detail["reason"] = "threshold_fail"
        return detail
    if not detail["margin_ok"]:
        detail["reason"] = "margin_fail"
        return detail
    if not detail["ev_ok"]:
        detail["reason"] = "ev_fail"
        return detail
    if not detail["score_ok"]:
        detail["reason"] = "score_fail"
        return detail

    detail["passed"] = True
    detail["reason"] = "passed_all"
    return detail


def format_heartbeat(row: pd.Series, long_d: Dict[str, Any], short_d: Dict[str, Any],
                     decision: str, decision_reason: str, size: float) -> str:
    ts = pd.to_datetime(row["timestamp"])
    close = float(row["close"])
    regime = str(row["regime_label"])

    lines = [
        f"[{ts}] close={close:.2f} regime={regime}",
        "LONG:"
        f" candidate={long_d['candidate']}"
        f" selected_regime={long_d['selected_regime']}"
        f" prob={long_d['prob']:.4f}"
        f" ev={long_d['ev']:.5f}"
        f" score={long_d['score']:.4f}"
        f" thr_ok={long_d['threshold_ok']}"
        f" margin_ok={long_d['margin_ok']}"
        f" ev_ok={long_d['ev_ok']}"
        f" score_ok={long_d['score_ok']}"
        f" reason={long_d['reason']}",
        "SHORT:"
        f" candidate={short_d['candidate']}"
        f" selected_regime={short_d['selected_regime']}"
        f" prob={short_d['prob']:.4f}"
        f" ev={short_d['ev']:.5f}"
        f" score={short_d['score']:.4f}"
        f" thr_ok={short_d['threshold_ok']}"
        f" margin_ok={short_d['margin_ok']}"
        f" ev_ok={short_d['ev_ok']}"
        f" score_ok={short_d['score_ok']}"
        f" reason={short_d['reason']}",
        f"DECISION: selected={decision} size={size:.4f} reason={decision_reason}",
    ]
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    CONFIG["fee_rate"] = args.fee
    CONFIG["slippage_bps"] = args.slippage

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 100)
    logger.info("PAPER TRACE SIMULATOR - CHAMPION ACTUAL")
    logger.info("=" * 100)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Fee: {CONFIG['fee_rate']}")
    logger.info(f"Slippage bps: {CONFIG['slippage_bps']}")
    logger.info(f"Bars to simulate: {args.bars}")

    long_model_path = MODEL_DIR / "meta_model_long_v3.pt"
    short_model_path = MODEL_DIR / "meta_model_short_v3.pt"
    long_scaler_path = SCALER_DIR / "meta_model_long_v3_scaler.json"
    short_scaler_path = SCALER_DIR / "meta_model_short_v3_scaler.json"

    for p in [DATA_PATH, long_model_path, short_model_path, long_scaler_path, short_scaler_path]:
        if not p.exists():
            raise FileNotFoundError(f"No existe {p}")

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["global_idx"] = np.arange(len(df), dtype=np.int64)

    if args.bars >= len(df):
        df_sim = df.copy().reset_index(drop=True)
    else:
        df_sim = df.iloc[-args.bars:].copy().reset_index(drop=True)

    long_scaler = load_scaler_payload(long_scaler_path)
    short_scaler = load_scaler_payload(short_scaler_path)

    long_feature_cols = long_scaler["feature_columns"]
    short_feature_cols = short_scaler["feature_columns"]

    X_long = apply_robust_scaler(df_sim, long_feature_cols, long_scaler)
    X_short = apply_robust_scaler(df_sim, short_feature_cols, short_scaler)

    long_model = MetaModel(input_dim=len(long_feature_cols)).to(DEVICE)
    short_model = MetaModel(input_dim=len(short_feature_cols)).to(DEVICE)

    long_model.load_state_dict(torch.load(long_model_path, map_location=DEVICE))
    short_model.load_state_dict(torch.load(short_model_path, map_location=DEVICE))

    long_model.eval()
    short_model.eval()

    with torch.no_grad():
        prob_long = torch.sigmoid(
            long_model(torch.tensor(X_long, dtype=torch.float32, device=DEVICE))
        ).cpu().numpy().flatten()

        prob_short = torch.sigmoid(
            short_model(torch.tensor(X_short, dtype=torch.float32, device=DEVICE))
        ).cpu().numpy().flatten()

    df_bt = df_sim.copy()
    df_bt["prob_long"] = prob_long
    df_bt["prob_short"] = prob_short

    atr = df_bt["atr"].replace(0, np.nan).ffill().bfill()
    close = df_bt["close"].replace(0, np.nan).ffill().bfill()

    df_bt["tp_long_pct"] = (CONFIG["tp_atr_mult_long"] * atr / close).clip(lower=0.0)
    df_bt["sl_long_pct"] = (CONFIG["sl_atr_mult_long"] * atr / close).clip(lower=0.0)
    df_bt["tp_short_pct"] = (CONFIG["tp_atr_mult_short"] * atr / close).clip(lower=0.0)
    df_bt["sl_short_pct"] = (CONFIG["sl_atr_mult_short"] * atr / close).clip(lower=0.0)

    df_bt = compute_setup_scores(df_bt)

    roundtrip_cost_pct = 2 * (CONFIG["fee_rate"] + CONFIG["slippage_bps"] / 10000.0)
    df_bt = compute_expected_values(df_bt, roundtrip_cost_pct=roundtrip_cost_pct)

    df_bt = infer_market_regime(df_bt)
    df_bt = apply_regime_adjustments(df_bt)

    df_bt = select_regime_ev_trades(
        df_bt, side="long",
        top_percent=CONFIG["top_percent_long"],
        min_ev_regime=CONFIG["min_ev_regime"]
    )
    df_bt = select_regime_ev_trades(
        df_bt, side="short",
        top_percent=CONFIG["top_percent_short"],
        min_ev_regime=CONFIG["min_ev_regime"]
    )

    capital = CONFIG["initial_capital"]
    equity_curve = []
    trade_log = []
    strategy_returns = []
    decision_trace = []

    i = 0
    cooldown = 0

    reason_counter = {}
    long_reason_counter = {}
    short_reason_counter = {}

    while i < len(df_bt) - 1:
        row = df_bt.iloc[i]

        if cooldown > 0:
            decision_reason = "cooldown_active"
            trace_row = {
                "timestamp": str(row["timestamp"]),
                "close": float(row["close"]),
                "regime_label": str(row["regime_label"]),
                "candidate_long": int(row.get("candidate_long", 0)),
                "candidate_short": int(row.get("candidate_short", 0)),
                "prob_long": float(row["prob_long"]),
                "prob_short": float(row["prob_short"]),
                "ev_long_regime": float(row["ev_long_regime"]),
                "ev_short_regime": float(row["ev_short_regime"]),
                "setup_score_long_regime": float(row["setup_score_long_regime"]),
                "setup_score_short_regime": float(row["setup_score_short_regime"]),
                "selected_long_regime": int(row.get("selected_long_regime", 0)),
                "selected_short_regime": int(row.get("selected_short_regime", 0)),
                "long_reason": "cooldown_active",
                "short_reason": "cooldown_active",
                "decision": "NONE",
                "decision_reason": decision_reason,
                "size": 0.0,
                "cooldown_active": 1,
            }
            decision_trace.append(trace_row)
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            reason_counter[decision_reason] = reason_counter.get(decision_reason, 0) + 1

            if not args.quiet:
                print(f"[{row['timestamp']}] cooldown_active | no trade")

            cooldown -= 1
            i += 1
            continue

        long_d = evaluate_long_detail(row)
        short_d = evaluate_short_detail(row)

        long_reason_counter[long_d["reason"]] = long_reason_counter.get(long_d["reason"], 0) + 1
        short_reason_counter[short_d["reason"]] = short_reason_counter.get(short_d["reason"], 0) + 1

        decision = "NONE"
        decision_reason = "no_side_passed"
        size = 0.0

        take_long = bool(long_d["passed"])
        take_short = bool(short_d["passed"])

        if take_long and take_short:
            rank_long = conflict_rank("long", row)
            rank_short = conflict_rank("short", row)
            if rank_long >= rank_short:
                take_short = False
                decision_reason = "long_won_conflict"
            else:
                take_long = False
                decision_reason = "short_won_conflict"

        if take_long:
            decision = "LONG"
            threshold = CONFIG["long_threshold"]
            size = compute_side_size(
                direction="long",
                prob=float(row["prob_long"]),
                threshold=threshold,
                setup_score=float(row["setup_score_long_regime"]),
                ev_used=float(row["ev_long_regime"]),
            )
            if decision_reason == "no_side_passed":
                decision_reason = "passed_all"

        elif take_short:
            decision = "SHORT"
            regime = str(row["regime_label"])
            threshold = (
                CONFIG["short_threshold_trend_down"]
                if regime == "trend_down"
                else CONFIG["short_threshold_range"]
            )
            size = compute_side_size(
                direction="short",
                prob=float(row["prob_short"]),
                threshold=threshold,
                setup_score=float(row["setup_score_short_regime"]),
                ev_used=float(row["ev_short_regime"]),
            )
            if decision_reason == "no_side_passed":
                decision_reason = "passed_all"

        trace_row = {
            "timestamp": str(row["timestamp"]),
            "close": float(row["close"]),
            "regime_label": str(row["regime_label"]),

            "candidate_long": long_d["candidate"],
            "selected_long_regime": long_d["selected_regime"],
            "prob_long": long_d["prob"],
            "ev_long_regime": long_d["ev"],
            "setup_score_long_regime": long_d["score"],
            "long_threshold_ok": long_d["threshold_ok"],
            "long_margin_ok": long_d["margin_ok"],
            "long_ev_ok": long_d["ev_ok"],
            "long_score_ok": long_d["score_ok"],
            "long_reason": long_d["reason"],

            "candidate_short": short_d["candidate"],
            "selected_short_regime": short_d["selected_regime"],
            "prob_short": short_d["prob"],
            "ev_short_regime": short_d["ev"],
            "setup_score_short_regime": short_d["score"],
            "short_threshold_ok": short_d["threshold_ok"],
            "short_margin_ok": short_d["margin_ok"],
            "short_ev_ok": short_d["ev_ok"],
            "short_score_ok": short_d["score_ok"],
            "short_reason": short_d["reason"],

            "decision": decision,
            "decision_reason": decision_reason,
            "size": size,
            "cooldown_active": 0,
        }
        decision_trace.append(trace_row)
        reason_counter[decision_reason] = reason_counter.get(decision_reason, 0) + 1

        if not args.quiet:
            print(format_heartbeat(row, long_d, short_d, decision, decision_reason, size))
            print("-" * 120)

        if decision == "NONE":
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            i += 1
            continue

        direction = "long" if decision == "LONG" else "short"
        gross_ret, exit_idx, outcome = simulate_trade(df_bt, i, direction)

        fee_cost = cost_for_transition(0.0, size, CONFIG["fee_rate"], CONFIG["slippage_bps"])
        fee_cost += cost_for_transition(size, 0.0, CONFIG["fee_rate"], CONFIG["slippage_bps"])

        net_ret = size * gross_ret - fee_cost
        capital *= (1.0 + net_ret)

        trade_log.append({
            "entry_idx": i,
            "exit_idx": exit_idx,
            "timestamp_entry": str(df_bt.iloc[i]["timestamp"]),
            "timestamp_exit": str(df_bt.iloc[min(exit_idx, len(df_bt)-1)]["timestamp"]),
            "direction": direction,
            "regime_label": row["regime_label"],
            "prob_long": float(row["prob_long"]),
            "prob_short": float(row["prob_short"]),
            "setup_score_long_regime": float(row["setup_score_long_regime"]),
            "setup_score_short_regime": float(row["setup_score_short_regime"]),
            "ev_long_regime": float(row["ev_long_regime"]),
            "ev_short_regime": float(row["ev_short_regime"]),
            "size": size,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "outcome": outcome,
            "decision_reason": decision_reason,
        })

        strategy_returns.append(net_ret)
        equity_curve.append(capital)
        i = max(i + 1, exit_idx + 1)
        cooldown = CONFIG["cooldown_bars"]

    decision_df = pd.DataFrame(decision_trace)
    trades_df = pd.DataFrame(trade_log)

    equity_curve = np.array(equity_curve, dtype=np.float64)
    strategy_returns = np.array(strategy_returns, dtype=np.float64)

    summary = {
        "bars_simulated": int(len(df_bt)),
        "initial_capital": CONFIG["initial_capital"],
        "final_capital": float(capital),
        "total_return": float(capital / CONFIG["initial_capital"] - 1.0),
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "sortino_ratio": sortino_ratio(strategy_returns),
        "n_trades": int(len(trades_df)),
        "avg_trade_return": float(trades_df["net_ret"].mean()) if len(trades_df) else 0.0,
        "win_rate_trade": float((trades_df["net_ret"] > 0).mean()) if len(trades_df) else 0.0,
        "long_trades": int((trades_df["direction"] == "long").sum()) if len(trades_df) else 0,
        "short_trades": int((trades_df["direction"] == "short").sum()) if len(trades_df) else 0,
        "avg_size": float(trades_df["size"].mean()) if len(trades_df) else 0.0,
        "reason_counter": reason_counter,
        "long_reason_counter": long_reason_counter,
        "short_reason_counter": short_reason_counter,
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

    logger.info("=" * 100)
    logger.info("PAPER TRACE SUMMARY")
    logger.info("=" * 100)
    for k, v in summary.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
