from __future__ import annotations

import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from src.strategy.setup_ranking import compute_setup_scores
from src.strategy.expected_value_engine import compute_expected_values
from src.strategy.regime_engine import infer_market_regime, apply_regime_adjustments


DATA_PATH = root_dir / "data/processed/dataset_btc_triple_barrier_1h.csv"
MODEL_DIR = root_dir / "artifacts/models"
SCALER_DIR = root_dir / "artifacts/scalers"
REPORT_DIR = root_dir / "artifacts/reports"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "threshold_long": 0.50,
    "threshold_short": 0.50,
    "prob_margin": 0.00,
    "min_ev_regime": 0.00005,
    "top_percent_long": 0.60,
    "top_percent_short": 0.60,
    "tp_atr_mult_long": 1.8,
    "sl_atr_mult_long": 1.2,
    "tp_atr_mult_short": 1.8,
    "sl_atr_mult_short": 1.2,
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,
}


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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


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


def build_test_split(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    train_end = int(n * 0.80)
    return df.iloc[train_end:].copy().reset_index(drop=True)


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
        out[f"{flag_col}_threshold"] = thr
    else:
        out[f"{flag_col}_threshold"] = np.nan

    return out


def main():
    logger.info("=" * 80)
    logger.info("DEBUG TRADE FUNNEL")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")

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
    df_test = build_test_split(df)

    long_scaler = load_scaler_payload(long_scaler_path)
    short_scaler = load_scaler_payload(short_scaler_path)

    long_feature_cols = long_scaler["feature_columns"]
    short_feature_cols = short_scaler["feature_columns"]

    X_long = apply_robust_scaler(df_test, long_feature_cols, long_scaler)
    X_short = apply_robust_scaler(df_test, short_feature_cols, short_scaler)

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

    df_bt = df_test.copy()
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

    # funnel long
    long_candidates = (df_bt["candidate_long"] == 1)
    long_prob = long_candidates & (df_bt["prob_long"] >= CONFIG["threshold_long"])
    long_margin = long_prob & ((df_bt["prob_long"] - df_bt["prob_short"]) >= CONFIG["prob_margin"])
    long_ev = long_margin & (df_bt["ev_long_regime"] > CONFIG["min_ev_regime"])
    long_selected = long_ev & (df_bt["selected_long_regime"] == 1)

    # funnel short
    short_candidates = (df_bt["candidate_short"] == 1)
    short_prob = short_candidates & (df_bt["prob_short"] >= CONFIG["threshold_short"])
    short_margin = short_prob & ((df_bt["prob_short"] - df_bt["prob_long"]) >= CONFIG["prob_margin"])
    short_ev = short_margin & (df_bt["ev_short_regime"] > CONFIG["min_ev_regime"])
    short_selected = short_ev & (df_bt["selected_short_regime"] == 1)

    summary = {
        "rows_test": int(len(df_bt)),
        "long": {
            "candidates": int(long_candidates.sum()),
            "prob_pass": int(long_prob.sum()),
            "margin_pass": int(long_margin.sum()),
            "ev_pass": int(long_ev.sum()),
            "selected_final": int(long_selected.sum()),
            "avg_prob_candidate": float(df_bt.loc[long_candidates, "prob_long"].mean()) if long_candidates.any() else 0.0,
            "avg_ev_candidate": float(df_bt.loc[long_candidates, "ev_long_regime"].mean()) if long_candidates.any() else 0.0,
            "selection_threshold_ev": float(df_bt["selected_long_regime_threshold"].iloc[0]) if "selected_long_regime_threshold" in df_bt.columns else np.nan,
            "regime_breakdown": df_bt.loc[long_candidates, "regime_label"].value_counts(normalize=True).to_dict() if long_candidates.any() else {},
        },
        "short": {
            "candidates": int(short_candidates.sum()),
            "prob_pass": int(short_prob.sum()),
            "margin_pass": int(short_margin.sum()),
            "ev_pass": int(short_ev.sum()),
            "selected_final": int(short_selected.sum()),
            "avg_prob_candidate": float(df_bt.loc[short_candidates, "prob_short"].mean()) if short_candidates.any() else 0.0,
            "avg_ev_candidate": float(df_bt.loc[short_candidates, "ev_short_regime"].mean()) if short_candidates.any() else 0.0,
            "selection_threshold_ev": float(df_bt["selected_short_regime_threshold"].iloc[0]) if "selected_short_regime_threshold" in df_bt.columns else np.nan,
            "regime_breakdown": df_bt.loc[short_candidates, "regime_label"].value_counts(normalize=True).to_dict() if short_candidates.any() else {},
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "debug_trade_funnel.json"
    out_csv = REPORT_DIR / "debug_trade_funnel_rows.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    debug_rows = df_bt[[
        "timestamp",
        "candidate_long", "candidate_short",
        "prob_long", "prob_short",
        "setup_score_long", "setup_score_short",
        "ev_long_net", "ev_short_net",
        "ev_long_regime", "ev_short_regime",
        "regime_label",
        "selected_long_regime", "selected_short_regime",
    ]].copy()
    debug_rows.to_csv(out_csv, index=False)

    logger.info("=" * 80)
    logger.info("FUNNEL SUMMARY")
    logger.info("=" * 80)
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
