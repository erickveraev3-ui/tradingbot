from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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

DATA_PATH = root_dir / "data" / "processed" / "dataset_btc_triple_barrier_1h.csv"
MODEL_DIR = root_dir / "artifacts/models"
SCALER_DIR = root_dir / "artifacts/scalers"
OUT_DIR_DEFAULT = root_dir / "artifacts/reports/champion_prelive_engine"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# CHAMPION CONFIG
# =============================================================================

CONFIG = {
    "initial_capital": 10000.0,
    "fee_rate": 0.0008,
    "slippage_bps": 5.0,
    "cooldown_bars": 1,

    # selection by regime/EV
    "min_ev_regime": 0.002,
    "top_percent_long": 0.35,
    "top_percent_short": 0.40,

    # sizing base
    "base_size": 0.10,
    "max_size": 0.60,
    "prob_scale": 2.0,
    "score_scale": 0.20,

    # champion primary long
    "long_threshold": 0.555,
    "long_prob_margin": 0.05,
    "long_min_ev": 0.0055,
    "long_min_setup": 1.90,

    # champion primary short
    "short_threshold_trend_down": 0.525,
    "short_prob_margin_trend_down": 0.03,
    "short_min_ev_trend_down": 0.0038,
    "short_min_setup_trend_down": 1.35,

    "short_threshold_range": 0.54,
    "short_prob_margin_range": 0.04,
    "short_min_ev_range": 0.0048,
    "short_min_setup_range": 1.55,

    # same barriers as dataset
    "tp_atr_mult_long": 1.8,
    "sl_atr_mult_long": 1.2,
    "tp_atr_mult_short": 1.8,
    "sl_atr_mult_short": 1.2,
}

SECONDARY = {
    # secondary long only in trend_up
    "long_threshold": 0.525,
    "long_prob_margin": 0.02,
    "long_min_ev": 0.0030,
    "long_min_setup": 1.45,

    # secondary short only in range
    "short_threshold_range": 0.55,
    "short_prob_margin_range": 0.03,
    "short_min_ev_range": 0.0035,
    "short_min_setup_range": 1.40,
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
# ARGS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Champion pre-live engine unificado.")

    p.add_argument("--mode", choices=["replay", "paper"], default="replay")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--start", type=str, default="")
    p.add_argument("--end", type=str, default="")
    p.add_argument("--fee", type=float, default=CONFIG["fee_rate"])
    p.add_argument("--slippage", type=float, default=CONFIG["slippage_bps"])
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


# =============================================================================
# UTILS
# =============================================================================

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
# DECISION LOGIC
# =============================================================================

def evaluate_primary_long(row: pd.Series) -> Dict[str, Any]:
    regime = str(row["regime_label"])
    candidate = int(row.get("candidate_long", 0))
    selected_regime = int(row.get("selected_long_regime", 0))

    prob = float(row["prob_long"])
    opp_prob = float(row["prob_short"])
    ev = float(row["ev_long_regime"])
    score = float(row["setup_score_long_regime"])

    d = {
        "layer": "primary",
        "side": "long",
        "candidate": candidate,
        "selected_regime": selected_regime,
        "prob": prob,
        "opp_prob": opp_prob,
        "ev": ev,
        "score": score,
        "regime": regime,
        "passed": False,
        "reason": "",
        "threshold_ok": 0,
        "margin_ok": 0,
        "ev_ok": 0,
        "score_ok": 0,
    }

    if regime != "trend_up":
        d["reason"] = "regime_block"
        return d
    if candidate != 1:
        d["reason"] = "no_candidate"
        return d
    if selected_regime != 1:
        d["reason"] = "not_selected_regime"
        return d

    d["threshold_ok"] = int(prob >= CONFIG["long_threshold"])
    d["margin_ok"] = int((prob - opp_prob) >= CONFIG["long_prob_margin"])
    d["ev_ok"] = int(ev >= CONFIG["long_min_ev"])
    d["score_ok"] = int(score >= CONFIG["long_min_setup"])

    if not d["threshold_ok"]:
        d["reason"] = "threshold_fail"
        return d
    if not d["margin_ok"]:
        d["reason"] = "margin_fail"
        return d
    if not d["ev_ok"]:
        d["reason"] = "ev_fail"
        return d
    if not d["score_ok"]:
        d["reason"] = "score_fail"
        return d

    d["passed"] = True
    d["reason"] = "passed_all"
    return d


def evaluate_primary_short(row: pd.Series) -> Dict[str, Any]:
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

    d = {
        "layer": "primary",
        "side": "short",
        "candidate": candidate,
        "selected_regime": selected_regime,
        "prob": prob,
        "opp_prob": opp_prob,
        "ev": ev,
        "score": score,
        "regime": regime,
        "passed": False,
        "reason": "",
        "threshold_ok": 0,
        "margin_ok": 0,
        "ev_ok": 0,
        "score_ok": 0,
    }

    if regime not in ("trend_down", "range"):
        d["reason"] = "regime_block"
        return d
    if candidate != 1:
        d["reason"] = "no_candidate"
        return d
    if selected_regime != 1:
        d["reason"] = "not_selected_regime"
        return d

    d["threshold_ok"] = int(prob >= threshold)
    d["margin_ok"] = int((prob - opp_prob) >= prob_margin)
    d["ev_ok"] = int(ev >= min_ev)
    d["score_ok"] = int(score >= min_setup)

    if not d["threshold_ok"]:
        d["reason"] = "threshold_fail"
        return d
    if not d["margin_ok"]:
        d["reason"] = "margin_fail"
        return d
    if not d["ev_ok"]:
        d["reason"] = "ev_fail"
        return d
    if not d["score_ok"]:
        d["reason"] = "score_fail"
        return d

    d["passed"] = True
    d["reason"] = "passed_all"
    return d


def evaluate_secondary_long(row: pd.Series) -> Dict[str, Any]:
    regime = str(row["regime_label"])
    candidate = int(row.get("candidate_long", 0))
    prob = float(row["prob_long"])
    opp_prob = float(row["prob_short"])
    ev = float(row["ev_long_regime"])
    score = float(row["setup_score_long_regime"])

    d = {
        "layer": "secondary",
        "side": "long",
        "candidate": candidate,
        "selected_regime": 0,
        "prob": prob,
        "opp_prob": opp_prob,
        "ev": ev,
        "score": score,
        "regime": regime,
        "passed": False,
        "reason": "",
        "threshold_ok": 0,
        "margin_ok": 0,
        "ev_ok": 0,
        "score_ok": 0,
    }

    if regime != "trend_up":
        d["reason"] = "secondary_regime_block"
        return d
    if candidate != 1:
        d["reason"] = "secondary_no_candidate"
        return d

    d["threshold_ok"] = int(prob >= SECONDARY["long_threshold"])
    d["margin_ok"] = int((prob - opp_prob) >= SECONDARY["long_prob_margin"])
    d["ev_ok"] = int(ev >= SECONDARY["long_min_ev"])
    d["score_ok"] = int(score >= SECONDARY["long_min_setup"])

    if not d["threshold_ok"]:
        d["reason"] = "secondary_threshold_fail"
        return d
    if not d["margin_ok"]:
        d["reason"] = "secondary_margin_fail"
        return d
    if not d["ev_ok"]:
        d["reason"] = "secondary_ev_fail"
        return d
    if not d["score_ok"]:
        d["reason"] = "secondary_score_fail"
        return d

    d["passed"] = True
    d["reason"] = "secondary_passed_all"
    return d


def evaluate_secondary_short(row: pd.Series) -> Dict[str, Any]:
    regime = str(row["regime_label"])
    candidate = int(row.get("candidate_short", 0))
    prob = float(row["prob_short"])
    opp_prob = float(row["prob_long"])
    ev = float(row["ev_short_regime"])
    score = float(row["setup_score_short_regime"])

    d = {
        "layer": "secondary",
        "side": "short",
        "candidate": candidate,
        "selected_regime": 0,
        "prob": prob,
        "opp_prob": opp_prob,
        "ev": ev,
        "score": score,
        "regime": regime,
        "passed": False,
        "reason": "",
        "threshold_ok": 0,
        "margin_ok": 0,
        "ev_ok": 0,
        "score_ok": 0,
    }

    if regime != "range":
        d["reason"] = "secondary_regime_block"
        return d
    if candidate != 1:
        d["reason"] = "secondary_no_candidate"
        return d

    d["threshold_ok"] = int(prob >= SECONDARY["short_threshold_range"])
    d["margin_ok"] = int((prob - opp_prob) >= SECONDARY["short_prob_margin_range"])
    d["ev_ok"] = int(ev >= SECONDARY["short_min_ev_range"])
    d["score_ok"] = int(score >= SECONDARY["short_min_setup_range"])

    if not d["threshold_ok"]:
        d["reason"] = "secondary_threshold_fail"
        return d
    if not d["margin_ok"]:
        d["reason"] = "secondary_margin_fail"
        return d
    if not d["ev_ok"]:
        d["reason"] = "secondary_ev_fail"
        return d
    if not d["score_ok"]:
        d["reason"] = "secondary_score_fail"
        return d

    d["passed"] = True
    d["reason"] = "secondary_passed_all"
    return d


def conflict_rank_primary(direction: str, row: pd.Series) -> float:
    if direction == "short":
        p_short = float(row["prob_short"])
        ev_short = float(row["ev_short_regime"])
        score_short = float(row["setup_score_short_regime"])
        return 1.00 * ev_short + 0.18 * max(0.0, p_short - 0.53) + 0.03 * score_short

    p_long = float(row["prob_long"])
    p_short = float(row["prob_short"])
    ev_long = float(row["ev_long_regime"])
    score_long = float(row["setup_score_long_regime"])

    rank = 1.00 * ev_long + 0.14 * max(0.0, p_long - CONFIG["long_threshold"]) + 0.025 * score_long
    if p_short >= 0.53:
        rank -= 0.0015
    if float(row["ev_short_regime"]) >= 0.0045:
        rank -= 0.0015
    if (p_long - p_short) < 0.06:
        rank -= 0.0010
    return rank


# =============================================================================
# SIZER V2
# =============================================================================

def base_side_size(direction: str, prob: float, threshold: float, setup_score: float, ev_used: float) -> float:
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


def regime_multiplier(decision_source: str, direction: str, regime: str) -> float:
    if decision_source == "primary":
        if direction == "short" and regime == "trend_down":
            return 1.10
        if direction == "long" and regime == "trend_up":
            return 1.00
        if regime == "range":
            return 0.92
        return 0.95

    if decision_source == "secondary":
        if direction == "long" and regime == "trend_up":
            return 0.65
        if direction == "short" and regime == "range":
            return 0.50
        return 0.40

    return 1.0


def source_cap(decision_source: str) -> float:
    return 0.60 if decision_source == "primary" else 0.15


def source_floor(decision_source: str) -> float:
    return 0.10 if decision_source == "primary" else 0.04


def compute_sizer_v2(
    decision_source: str,
    direction: str,
    regime: str,
    prob: float,
    threshold: float,
    setup_score: float,
    ev_used: float,
) -> float:
    base = base_side_size(direction, prob, threshold, setup_score, ev_used)
    size = base * regime_multiplier(decision_source, direction, regime)
    size = min(source_cap(decision_source), size)
    size = max(source_floor(decision_source), size)
    return float(size)


# =============================================================================
# PREP DATA
# =============================================================================

def load_and_prepare_dataset(args) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["global_idx"] = np.arange(len(df), dtype=np.int64)

    if args.start:
        df = df[df["timestamp"] >= pd.to_datetime(args.start)].copy()
    if args.end:
        df = df[df["timestamp"] <= pd.to_datetime(args.end)].copy()

    if args.start or args.end:
        df = df.reset_index(drop=True)
    else:
        bars = min(args.bars, len(df))
        df = df.iloc[-bars:].copy().reset_index(drop=True)

    return df


def infer_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    long_model_path = MODEL_DIR / "meta_model_long_v3.pt"
    short_model_path = MODEL_DIR / "meta_model_short_v3.pt"
    long_scaler_path = SCALER_DIR / "meta_model_long_v3_scaler.json"
    short_scaler_path = SCALER_DIR / "meta_model_short_v3_scaler.json"

    for p in [long_model_path, short_model_path, long_scaler_path, short_scaler_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    long_scaler = load_scaler_payload(long_scaler_path)
    short_scaler = load_scaler_payload(short_scaler_path)

    X_long = apply_robust_scaler(df, long_scaler["feature_columns"], long_scaler)
    X_short = apply_robust_scaler(df, short_scaler["feature_columns"], short_scaler)

    long_model = MetaModel(input_dim=len(long_scaler["feature_columns"])).to(DEVICE)
    short_model = MetaModel(input_dim=len(short_scaler["feature_columns"])).to(DEVICE)

    long_model.load_state_dict(torch.load(long_model_path, map_location=DEVICE))
    short_model.load_state_dict(torch.load(short_model_path, map_location=DEVICE))
    long_model.eval()
    short_model.eval()

    with torch.no_grad():
        prob_long = torch.sigmoid(long_model(torch.tensor(X_long, dtype=torch.float32, device=DEVICE))).cpu().numpy().flatten()
        prob_short = torch.sigmoid(short_model(torch.tensor(X_short, dtype=torch.float32, device=DEVICE))).cpu().numpy().flatten()

    out = df.copy()
    out["prob_long"] = prob_long
    out["prob_short"] = prob_short

    atr = out["atr"].replace(0, np.nan).ffill().bfill()
    close = out["close"].replace(0, np.nan).ffill().bfill()

    out["tp_long_pct"] = (CONFIG["tp_atr_mult_long"] * atr / close).clip(lower=0.0)
    out["sl_long_pct"] = (CONFIG["sl_atr_mult_long"] * atr / close).clip(lower=0.0)
    out["tp_short_pct"] = (CONFIG["tp_atr_mult_short"] * atr / close).clip(lower=0.0)
    out["sl_short_pct"] = (CONFIG["sl_atr_mult_short"] * atr / close).clip(lower=0.0)

    out = compute_setup_scores(out)
    roundtrip_cost_pct = 2 * (CONFIG["fee_rate"] + CONFIG["slippage_bps"] / 10000.0)
    out = compute_expected_values(out, roundtrip_cost_pct=roundtrip_cost_pct)
    out = infer_market_regime(out)
    out = apply_regime_adjustments(out)

    out = select_regime_ev_trades(out, side="long", top_percent=CONFIG["top_percent_long"], min_ev_regime=CONFIG["min_ev_regime"])
    out = select_regime_ev_trades(out, side="short", top_percent=CONFIG["top_percent_short"], min_ev_regime=CONFIG["min_ev_regime"])

    return out


# =============================================================================
# ENGINE
# =============================================================================

def format_heartbeat(row: pd.Series, dpl: Dict[str, Any], dps: Dict[str, Any],
                     dsl: Dict[str, Any], dss: Dict[str, Any],
                     decision: str, source: str, reason: str, size: float) -> str:
    ts = pd.to_datetime(row["timestamp"])
    close = float(row["close"])
    regime = str(row["regime_label"])

    return "\n".join([
        f"[{ts}] close={close:.2f} regime={regime}",
        f"PRIMARY LONG: candidate={dpl['candidate']} sel={dpl['selected_regime']} prob={dpl['prob']:.4f} ev={dpl['ev']:.5f} score={dpl['score']:.4f} reason={dpl['reason']}",
        f"PRIMARY SHORT: candidate={dps['candidate']} sel={dps['selected_regime']} prob={dps['prob']:.4f} ev={dps['ev']:.5f} score={dps['score']:.4f} reason={dps['reason']}",
        f"SECONDARY LONG: candidate={dsl['candidate']} prob={dsl['prob']:.4f} ev={dsl['ev']:.5f} score={dsl['score']:.4f} reason={dsl['reason']}",
        f"SECONDARY SHORT: candidate={dss['candidate']} prob={dss['prob']:.4f} ev={dss['ev']:.5f} score={dss['score']:.4f} reason={dss['reason']}",
        f"DECISION: selected={decision} source={source} size={size:.4f} reason={reason}",
    ])


def run_engine(df_bt: pd.DataFrame, quiet: bool) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    capital = CONFIG["initial_capital"]
    equity_curve = []
    strategy_returns = []
    trade_log = []
    decision_trace = []

    counters = {
        "primary_passed": 0,
        "secondary_passed": 0,
        "no_trade": 0,
        "cooldown_active": 0,
    }

    decision_reason_counter = {}
    primary_long_reason_counter = {}
    primary_short_reason_counter = {}
    secondary_long_reason_counter = {}
    secondary_short_reason_counter = {}
    regime_counter = {}

    i = 0
    cooldown = 0

    while i < len(df_bt) - 1:
        row = df_bt.iloc[i]
        regime = str(row["regime_label"])
        regime_counter[regime] = regime_counter.get(regime, 0) + 1

        if cooldown > 0:
            decision_trace.append({
                "timestamp": str(row["timestamp"]),
                "close": float(row["close"]),
                "regime_label": regime,
                "decision": "NONE",
                "decision_source": "cooldown",
                "decision_reason": "cooldown_active",
                "size": 0.0,
            })
            counters["cooldown_active"] += 1
            decision_reason_counter["cooldown_active"] = decision_reason_counter.get("cooldown_active", 0) + 1
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            cooldown -= 1
            i += 1
            continue

        dpl = evaluate_primary_long(row)
        dps = evaluate_primary_short(row)
        dsl = evaluate_secondary_long(row)
        dss = evaluate_secondary_short(row)

        primary_long_reason_counter[dpl["reason"]] = primary_long_reason_counter.get(dpl["reason"], 0) + 1
        primary_short_reason_counter[dps["reason"]] = primary_short_reason_counter.get(dps["reason"], 0) + 1
        secondary_long_reason_counter[dsl["reason"]] = secondary_long_reason_counter.get(dsl["reason"], 0) + 1
        secondary_short_reason_counter[dss["reason"]] = secondary_short_reason_counter.get(dss["reason"], 0) + 1

        decision = "NONE"
        decision_source = "none"
        decision_reason = "no_side_passed"
        size = 0.0

        # PRIMARY LAYER
        take_pl = bool(dpl["passed"])
        take_ps = bool(dps["passed"])

        if take_pl and take_ps:
            rank_long = conflict_rank_primary("long", row)
            rank_short = conflict_rank_primary("short", row)
            if rank_long >= rank_short:
                take_ps = False
                decision_reason = "primary_long_won_conflict"
            else:
                take_pl = False
                decision_reason = "primary_short_won_conflict"

        if take_pl:
            decision = "LONG"
            decision_source = "primary"
            size = compute_sizer_v2(
                decision_source="primary",
                direction="long",
                regime=regime,
                prob=float(row["prob_long"]),
                threshold=CONFIG["long_threshold"],
                setup_score=float(row["setup_score_long_regime"]),
                ev_used=float(row["ev_long_regime"]),
            )
            decision_reason = "passed_all" if decision_reason == "no_side_passed" else decision_reason
            counters["primary_passed"] += 1

        elif take_ps:
            threshold = CONFIG["short_threshold_trend_down"] if regime == "trend_down" else CONFIG["short_threshold_range"]
            decision = "SHORT"
            decision_source = "primary"
            size = compute_sizer_v2(
                decision_source="primary",
                direction="short",
                regime=regime,
                prob=float(row["prob_short"]),
                threshold=threshold,
                setup_score=float(row["setup_score_short_regime"]),
                ev_used=float(row["ev_short_regime"]),
            )
            decision_reason = "passed_all" if decision_reason == "no_side_passed" else decision_reason
            counters["primary_passed"] += 1

        # SECONDARY LAYER ONLY IF PRIMARY DID NOTHING
        if decision == "NONE":
            take_sl = bool(dsl["passed"])
            take_ss = bool(dss["passed"])

            if take_sl and take_ss:
                if float(row["ev_long_regime"]) >= float(row["ev_short_regime"]):
                    take_ss = False
                    decision_reason = "secondary_long_won_conflict"
                else:
                    take_sl = False
                    decision_reason = "secondary_short_won_conflict"

            if take_sl:
                decision = "LONG"
                decision_source = "secondary"
                size = compute_sizer_v2(
                    decision_source="secondary",
                    direction="long",
                    regime=regime,
                    prob=float(row["prob_long"]),
                    threshold=SECONDARY["long_threshold"],
                    setup_score=float(row["setup_score_long_regime"]),
                    ev_used=float(row["ev_long_regime"]),
                )
                decision_reason = dsl["reason"]
                counters["secondary_passed"] += 1

            elif take_ss:
                decision = "SHORT"
                decision_source = "secondary"
                size = compute_sizer_v2(
                    decision_source="secondary",
                    direction="short",
                    regime=regime,
                    prob=float(row["prob_short"]),
                    threshold=SECONDARY["short_threshold_range"],
                    setup_score=float(row["setup_score_short_regime"]),
                    ev_used=float(row["ev_short_regime"]),
                )
                decision_reason = dss["reason"]
                counters["secondary_passed"] += 1

        if decision == "NONE":
            counters["no_trade"] += 1

        decision_trace.append({
            "timestamp": str(row["timestamp"]),
            "close": float(row["close"]),
            "regime_label": regime,

            "primary_long_reason": dpl["reason"],
            "primary_short_reason": dps["reason"],
            "secondary_long_reason": dsl["reason"],
            "secondary_short_reason": dss["reason"],

            "candidate_long": int(row.get("candidate_long", 0)),
            "candidate_short": int(row.get("candidate_short", 0)),
            "selected_long_regime": int(row.get("selected_long_regime", 0)),
            "selected_short_regime": int(row.get("selected_short_regime", 0)),

            "prob_long": float(row["prob_long"]),
            "prob_short": float(row["prob_short"]),
            "ev_long_regime": float(row["ev_long_regime"]),
            "ev_short_regime": float(row["ev_short_regime"]),
            "setup_score_long_regime": float(row["setup_score_long_regime"]),
            "setup_score_short_regime": float(row["setup_score_short_regime"]),

            "decision": decision,
            "decision_source": decision_source,
            "decision_reason": decision_reason,
            "size": size,
        })

        decision_reason_counter[decision_reason] = decision_reason_counter.get(decision_reason, 0) + 1

        if not quiet:
            print(format_heartbeat(row, dpl, dps, dsl, dss, decision, decision_source, decision_reason, size))
            print("-" * 140)

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

    summary = {
        "bars_processed": int(len(df_bt)),
        "start_timestamp": str(df_bt["timestamp"].iloc[0]) if len(df_bt) else "",
        "end_timestamp": str(df_bt["timestamp"].iloc[-1]) if len(df_bt) else "",
        "initial_capital": CONFIG["initial_capital"],
        "final_capital": float(capital),
        "total_return": float(capital / CONFIG["initial_capital"] - 1.0),
        "max_drawdown": max_drawdown(np.array(equity_curve, dtype=np.float64)),
        "sharpe_ratio": sharpe_ratio(np.array(strategy_returns, dtype=np.float64)),
        "sortino_ratio": sortino_ratio(np.array(strategy_returns, dtype=np.float64)),
        "n_trades": int(len(trade_log)),
        "avg_trade_return": float(pd.DataFrame(trade_log)["net_ret"].mean()) if trade_log else 0.0,
        "win_rate_trade": float((pd.DataFrame(trade_log)["net_ret"] > 0).mean()) if trade_log else 0.0,
        "long_trades": int((pd.DataFrame(trade_log)["direction"] == "long").sum()) if trade_log else 0,
        "short_trades": int((pd.DataFrame(trade_log)["direction"] == "short").sum()) if trade_log else 0,
        "primary_trades": int((pd.DataFrame(trade_log)["decision_source"] == "primary").sum()) if trade_log else 0,
        "secondary_trades": int((pd.DataFrame(trade_log)["decision_source"] == "secondary").sum()) if trade_log else 0,
        "avg_primary_size": float(pd.DataFrame(trade_log).loc[pd.DataFrame(trade_log)["decision_source"] == "primary", "size"].mean()) if trade_log else 0.0,
        "avg_secondary_size": float(pd.DataFrame(trade_log).loc[pd.DataFrame(trade_log)["decision_source"] == "secondary", "size"].mean()) if trade_log else 0.0,
        "decision_reason_counter": decision_reason_counter,
        "primary_long_reason_counter": primary_long_reason_counter,
        "primary_short_reason_counter": primary_short_reason_counter,
        "secondary_long_reason_counter": secondary_long_reason_counter,
        "secondary_short_reason_counter": secondary_short_reason_counter,
        "regime_counter": regime_counter,
        "counters": counters,
    }

    return (
        pd.DataFrame(decision_trace),
        pd.DataFrame(trade_log),
        np.array(equity_curve, dtype=np.float64),
        np.array(strategy_returns, dtype=np.float64),
        summary,
    )


# =============================================================================
# SAVE / MAIN
# =============================================================================

def save_outputs(out_dir: Path, decision_df: pd.DataFrame, trades_df: pd.DataFrame,
                 equity_curve: np.ndarray, strategy_returns: np.ndarray, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_df.to_csv(out_dir / "decision_trace.csv", index=False)
    trades_df.to_csv(out_dir / "simulated_trades.csv", index=False)

    pd.DataFrame({
        "step": np.arange(len(equity_curve)),
        "equity": equity_curve,
        "strategy_return": strategy_returns,
    }).to_csv(out_dir / "equity_curve.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    CONFIG["fee_rate"] = args.fee
    CONFIG["slippage_bps"] = args.slippage

    out_dir = Path(args.out_dir)
    logger.info("=" * 100)
    logger.info("CHAMPION PRELIVE ENGINE")
    logger.info("=" * 100)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Fee: {CONFIG['fee_rate']}")
    logger.info(f"Slippage bps: {CONFIG['slippage_bps']}")

    df = load_and_prepare_dataset(args)
    df_bt = infer_probabilities(df)

    decision_df, trades_df, equity_curve, strategy_returns, summary = run_engine(df_bt, quiet=args.quiet)
    save_outputs(out_dir, decision_df, trades_df, equity_curve, strategy_returns, summary)

    logger.info("=" * 100)
    logger.info("CHAMPION PRELIVE SUMMARY")
    logger.info("=" * 100)
    for k, v in summary.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
