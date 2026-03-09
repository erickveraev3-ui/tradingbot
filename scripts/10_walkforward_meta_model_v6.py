from __future__ import annotations

import sys
import json
import random
from dataclasses import dataclass
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger

from src.strategy.setup_ranking import compute_setup_scores
from src.strategy.position_sizer import dynamic_position_size
from src.strategy.expected_value_engine import compute_expected_values
from src.strategy.regime_engine import infer_market_regime, apply_regime_adjustments


# =============================================================================
# PATHS
# =============================================================================

DATA_PATH = root_dir / "data/processed/dataset_btc_triple_barrier_1h.csv"
OUT_DIR = root_dir / "artifacts/reports/walkforward_meta_model_v6_regime_aware"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # reproducibilidad
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # entrenamiento
    "batch_size": 256,
    "epochs": 50,
    "lr": 8e-4,
    "weight_decay": 1e-5,
    "dropout": 0.20,
    "early_stopping_patience": 7,

    # walk-forward
    "train_bars": 30000,
    "test_bars": 3000,
    "step_bars": 3000,
    "expanding_window": True,
    "min_train_candidate_rows_per_side": 250,

    # ejecución / costes
    "initial_capital": 10000.0,
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,
    "cooldown_bars": 1,

    # comparación
    "modes": ["dual", "long_only", "short_only"],

    # regímenes permitidos por lado
    "allowed_regimes_long": ["trend_up", "range"],
    "allowed_regimes_short": ["trend_down", "range"],

    # fallback si aparece un régimen no mapeado
    "default_long_threshold": 0.54,
    "default_long_prob_margin": 0.03,
    "default_long_min_ev": 0.0040,
    "default_long_min_setup": 1.50,

    "default_short_threshold": 0.54,
    "default_short_prob_margin": 0.04,
    "default_short_min_ev": 0.0045,
    "default_short_min_setup": 1.50,

    # filtros long por régimen
    "long_regime_params": {
        "trend_up": {
            "threshold": 0.515,
            "prob_margin": 0.02,
            "min_ev": 0.0030,
            "min_setup": 1.30,
            "top_percent": 0.35,
        },
        "range": {
            "threshold": 0.535,
            "prob_margin": 0.03,
            "min_ev": 0.0045,
            "min_setup": 1.55,
            "top_percent": 0.20,
        },
        "trend_down": {
            "threshold": 0.57,
            "prob_margin": 0.06,
            "min_ev": 0.0070,
            "min_setup": 2.20,
            "top_percent": 0.08,
        },
    },

    # filtros short por régimen
    "short_regime_params": {
        "trend_down": {
            "threshold": 0.525,
            "prob_margin": 0.03,
            "min_ev": 0.0038,
            "min_setup": 1.35,
            "top_percent": 0.32,
        },
        "range": {
            "threshold": 0.54,
            "prob_margin": 0.04,
            "min_ev": 0.0048,
            "min_setup": 1.55,
            "top_percent": 0.18,
        },
        "trend_up": {
            "threshold": 0.575,
            "prob_margin": 0.07,
            "min_ev": 0.0075,
            "min_setup": 2.25,
            "top_percent": 0.08,
        },
    },

    # sizing long
    "long_base_size": 0.05,
    "long_max_size": 0.22,
    "long_prob_scale": 1.10,
    "long_score_scale": 0.09,

    # sizing short
    "short_base_size": 0.05,
    "short_max_size": 0.25,
    "short_prob_scale": 1.25,
    "short_score_scale": 0.10,

    # conflicto dual
    "dual_conflict_ev_weight_long": 1.00,
    "dual_conflict_ev_weight_short": 1.00,
    "dual_conflict_prob_weight_long": 0.16,
    "dual_conflict_prob_weight_short": 0.18,
    "dual_conflict_score_weight_long": 0.03,
    "dual_conflict_score_weight_short": 0.03,

    # triple barrier proxy
    "tp_atr_mult_long": 1.8,
    "sl_atr_mult_long": 1.2,
    "tp_atr_mult_short": 1.8,
    "sl_atr_mult_short": 1.2,
}


# =============================================================================
# UTILS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


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


def cost_for_transition(pos_prev: float, pos_new: float, fee_rate: float, slippage_bps: float) -> float:
    turnover = abs(pos_new - pos_prev)
    if turnover == 0:
        return 0.0
    return turnover * fee_rate + turnover * (slippage_bps / 10000.0)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = []
    for c in df.columns:
        if c == "timestamp":
            continue
        if c in ["open", "high", "low", "close", "volume", "quote_volume", "trades"]:
            continue
        if c.startswith("tb_"):
            continue
        if c.startswith("event_"):
            continue
        if c.startswith("future_"):
            continue
        if c.startswith("target_"):
            continue
        if c in ["candidate_long", "candidate_short", "global_idx"]:
            continue
        feature_cols.append(c)
    return feature_cols


def build_folds(
    n_rows: int,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    expanding_window: bool = True,
) -> list[dict]:
    folds = []
    test_start = train_bars
    fold_id = 0

    while test_start + test_bars <= n_rows:
        train_end = test_start
        train_start = 0 if expanding_window else max(0, train_end - train_bars)
        test_end = test_start + test_bars

        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        fold_id += 1
        test_start += step_bars

    return folds


def bucketize_series(s: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=True)


def get_long_regime_params(regime: str) -> dict:
    return CONFIG["long_regime_params"].get(
        regime,
        {
            "threshold": CONFIG["default_long_threshold"],
            "prob_margin": CONFIG["default_long_prob_margin"],
            "min_ev": CONFIG["default_long_min_ev"],
            "min_setup": CONFIG["default_long_min_setup"],
            "top_percent": 0.15,
        },
    )


def get_short_regime_params(regime: str) -> dict:
    return CONFIG["short_regime_params"].get(
        regime,
        {
            "threshold": CONFIG["default_short_threshold"],
            "prob_margin": CONFIG["default_short_prob_margin"],
            "min_ev": CONFIG["default_short_min_ev"],
            "min_setup": CONFIG["default_short_min_setup"],
            "top_percent": 0.15,
        },
    )


# =============================================================================
# MODEL
# =============================================================================

class TradingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class MetaModel(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.20):
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


@dataclass
class SideModelBundle:
    model: MetaModel | None
    scaler: RobustScaler | None
    feature_cols: list[str]
    stats: dict


def train_one_side_fold(
    df_train: pd.DataFrame,
    candidate_col: str,
    label_col: str,
    side_name: str,
) -> SideModelBundle:
    device = CONFIG["device"]
    df_side = df_train[df_train[candidate_col] == 1].copy().reset_index(drop=True)
    fallback_feature_cols = select_feature_columns(df_train)

    if len(df_side) < CONFIG["min_train_candidate_rows_per_side"]:
        logger.warning(f"[{side_name}] too few candidate rows: {len(df_side)}")
        return SideModelBundle(
            model=None,
            scaler=None,
            feature_cols=fallback_feature_cols,
            stats={
                "rows": int(len(df_side)),
                "positive_rate_total": float("nan"),
                "positive_rate_train": float("nan"),
                "best_val_ap": float("nan"),
                "trained": False,
                "reason": "too_few_candidate_rows",
            },
        )

    y = (df_side[label_col] == 1).astype(np.float32).values
    feature_cols = select_feature_columns(df_side)

    if len(feature_cols) == 0:
        raise ValueError(f"[{side_name}] no features after filtering")

    if len(np.unique(y)) < 2:
        logger.warning(f"[{side_name}] single class in train")
        return SideModelBundle(
            model=None,
            scaler=None,
            feature_cols=feature_cols,
            stats={
                "rows": int(len(df_side)),
                "positive_rate_total": float(np.mean(y)),
                "positive_rate_train": float(np.mean(y)),
                "best_val_ap": float("nan"),
                "trained": False,
                "reason": "single_class_train",
            },
        )

    X = df_side[feature_cols].values.astype(np.float32)

    split = int(len(X) * 0.8)
    split = max(split, 1)
    split = min(split, len(X) - 1)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = np.clip(np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0), -8, 8)
    X_val = np.clip(np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0), -8, 8)

    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = neg_count / max(pos_count, 1.0)

    train_ds = TradingDataset(X_train, y_train)
    val_ds = TradingDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = MetaModel(input_dim=X.shape[1], dropout=CONFIG["dropout"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    best_ap = -np.inf
    best_state = None
    patience = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy().flatten()
                preds.extend(prob.tolist())
                trues.extend(yb.numpy().tolist())

        preds = np.array(preds, dtype=np.float32)
        trues = np.array(trues, dtype=np.float32)

        ap = safe_ap(trues, preds)
        auc = safe_auc(trues, preds)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        logger.info(
            f"[{side_name}] epoch={epoch:02d} "
            f"loss={train_loss:.4f} auc={auc:.4f} ap={ap:.4f}"
        )

        metric_for_es = ap if not np.isnan(ap) else -np.inf
        if metric_for_es > best_ap:
            best_ap = metric_for_es
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                logger.info(f"[{side_name}] early stopping")
                break

    if best_state is None:
        return SideModelBundle(
            model=None,
            scaler=None,
            feature_cols=feature_cols,
            stats={
                "rows": int(len(df_side)),
                "positive_rate_total": float(np.mean(y)),
                "positive_rate_train": float(np.mean(y_train)),
                "best_val_ap": float("nan"),
                "trained": False,
                "reason": "no_best_state",
            },
        )

    model.load_state_dict(best_state)
    model.eval()

    return SideModelBundle(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        stats={
            "rows": int(len(df_side)),
            "positive_rate_total": float(np.mean(y)),
            "positive_rate_train": float(np.mean(y_train)),
            "best_val_ap": float(best_ap),
            "trained": True,
            "reason": "ok",
        },
    )


def predict_side_bundle(bundle: SideModelBundle, df_test: pd.DataFrame) -> np.ndarray:
    if len(df_test) == 0:
        return np.array([], dtype=np.float32)

    if bundle.model is None or bundle.scaler is None:
        return np.full(len(df_test), 0.50, dtype=np.float32)

    missing = [c for c in bundle.feature_cols if c not in df_test.columns]
    if missing:
        raise ValueError(f"Missing columns in test: {missing[:10]}")

    X = df_test[bundle.feature_cols].values.astype(np.float32)
    X = bundle.scaler.transform(X)
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -8, 8)

    device = CONFIG["device"]
    with torch.no_grad():
        probs = torch.sigmoid(
            bundle.model(torch.tensor(X, dtype=torch.float32, device=device))
        ).cpu().numpy().flatten()

    return probs.astype(np.float32)


# =============================================================================
# PREP / SELECTION
# =============================================================================

def select_regime_ev_trades(df: pd.DataFrame, side: str) -> pd.DataFrame:
    out = df.copy()

    if side == "long":
        ev_col = "ev_long_regime"
        setup_col = "setup_score_long_regime"
        candidate_col = "candidate_long"
        regime_col = "regime_label"
        flag_col = "selected_long_regime"
        params_map = CONFIG["long_regime_params"]
    elif side == "short":
        ev_col = "ev_short_regime"
        setup_col = "setup_score_short_regime"
        candidate_col = "candidate_short"
        regime_col = "regime_label"
        flag_col = "selected_short_regime"
        params_map = CONFIG["short_regime_params"]
    else:
        raise ValueError("side must be 'long' or 'short'")

    out[flag_col] = 0

    for regime_name, params in params_map.items():
        mask = (
            (out[candidate_col] == 1) &
            (out[regime_col] == regime_name) &
            (out[ev_col] >= params["min_ev"]) &
            (out[setup_col] >= params["min_setup"])
        )
        if mask.any():
            thr = out.loc[mask, ev_col].quantile(1.0 - params["top_percent"])
            out.loc[mask & (out[ev_col] >= thr), flag_col] = 1

    return out


def prepare_test_frame(df_test: pd.DataFrame, prob_long: np.ndarray, prob_short: np.ndarray) -> pd.DataFrame:
    out = df_test.copy()
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

    out = select_regime_ev_trades(out, side="long")
    out = select_regime_ev_trades(out, side="short")

    return out


# =============================================================================
# TRADE SIM
# =============================================================================

def simulate_trade_within_fold(
    df_fold: pd.DataFrame,
    entry_idx_local: int,
    direction: str,
    fold_global_last: int,
    global_to_local: dict[int, int],
):
    row = df_fold.iloc[entry_idx_local]
    entry_global_idx = int(row["global_idx"])

    if direction == "long":
        label = int(row["tb_long_label"])
        gross_ret = float(row["tb_long_return"])
        hit_bar_global = int(row["tb_long_hit_bar"])
    else:
        label = int(row["tb_short_label"])
        gross_ret = float(row["tb_short_return"])
        hit_bar_global = int(row["tb_short_hit_bar"])

    if hit_bar_global < entry_global_idx + 1:
        hit_bar_global = entry_global_idx + 1

    if hit_bar_global > fold_global_last:
        return None

    if hit_bar_global not in global_to_local:
        return None

    exit_idx_local = global_to_local[hit_bar_global]

    if label == 1:
        outcome = "tp"
    elif label == -1:
        outcome = "sl"
    else:
        outcome = "expiry"

    return gross_ret, exit_idx_local, outcome


def compute_side_size(direction: str, prob: float, threshold: float, setup_score: float, ev_used: float) -> float:
    if direction == "long":
        base_size = CONFIG["long_base_size"]
        max_size = CONFIG["long_max_size"]
        prob_scale = CONFIG["long_prob_scale"]
        score_scale = CONFIG["long_score_scale"]
        score_input = setup_score + max(0.0, 28.0 * ev_used)
    else:
        base_size = CONFIG["short_base_size"]
        max_size = CONFIG["short_max_size"]
        prob_scale = CONFIG["short_prob_scale"]
        score_scale = CONFIG["short_score_scale"]
        score_input = setup_score + max(0.0, 40.0 * ev_used)

    size = dynamic_position_size(
        prob=prob,
        threshold=threshold,
        setup_score=score_input,
        base_size=base_size,
        max_size=max_size,
        prob_scale=prob_scale,
        score_scale=score_scale,
    )
    return float(min(max_size, max(base_size, size)))


def evaluate_long_pass(row: pd.Series, mode: str) -> tuple[bool, dict]:
    regime = str(row["regime_label"])
    params = get_long_regime_params(regime)

    if mode == "short_only":
        return False, {"reason": "mode_block"}

    if regime not in CONFIG["allowed_regimes_long"]:
        return False, {"reason": "regime_block", "regime": regime}

    p_long = float(row["prob_long"])
    p_short = float(row["prob_short"])
    ev_long = float(row["ev_long_regime"])
    score_long = float(row["setup_score_long_regime"])
    selected = int(row["selected_long_regime"]) == 1

    passed = (
        selected and
        p_long >= params["threshold"] and
        (p_long - p_short) >= params["prob_margin"] and
        ev_long >= params["min_ev"] and
        score_long >= params["min_setup"]
    )

    info = {
        "regime": regime,
        "threshold": params["threshold"],
        "prob_margin": params["prob_margin"],
        "min_ev": params["min_ev"],
        "min_setup": params["min_setup"],
        "selected_flag": int(selected),
    }
    return bool(passed), info


def evaluate_short_pass(row: pd.Series, mode: str) -> tuple[bool, dict]:
    regime = str(row["regime_label"])
    params = get_short_regime_params(regime)

    if mode == "long_only":
        return False, {"reason": "mode_block"}

    if regime not in CONFIG["allowed_regimes_short"]:
        return False, {"reason": "regime_block", "regime": regime}

    p_long = float(row["prob_long"])
    p_short = float(row["prob_short"])
    ev_short = float(row["ev_short_regime"])
    score_short = float(row["setup_score_short_regime"])
    selected = int(row["selected_short_regime"]) == 1

    passed = (
        selected and
        p_short >= params["threshold"] and
        (p_short - p_long) >= params["prob_margin"] and
        ev_short >= params["min_ev"] and
        score_short >= params["min_setup"]
    )

    info = {
        "regime": regime,
        "threshold": params["threshold"],
        "prob_margin": params["prob_margin"],
        "min_ev": params["min_ev"],
        "min_setup": params["min_setup"],
        "selected_flag": int(selected),
    }
    return bool(passed), info


def dual_conflict_rank(direction: str, row: pd.Series) -> float:
    if direction == "long":
        regime = str(row["regime_label"])
        params = get_long_regime_params(regime)
        ev_used = float(row["ev_long_regime"])
        prob_edge = float(row["prob_long"] - params["threshold"])
        score_used = float(row["setup_score_long_regime"])
        return (
            CONFIG["dual_conflict_ev_weight_long"] * ev_used +
            CONFIG["dual_conflict_prob_weight_long"] * max(0.0, prob_edge) +
            CONFIG["dual_conflict_score_weight_long"] * score_used
        )

    regime = str(row["regime_label"])
    params = get_short_regime_params(regime)
    ev_used = float(row["ev_short_regime"])
    prob_edge = float(row["prob_short"] - params["threshold"])
    score_used = float(row["setup_score_short_regime"])
    return (
        CONFIG["dual_conflict_ev_weight_short"] * ev_used +
        CONFIG["dual_conflict_prob_weight_short"] * max(0.0, prob_edge) +
        CONFIG["dual_conflict_score_weight_short"] * score_used
    )


def run_backtest_on_fold(
    df_fold: pd.DataFrame,
    capital_in: float,
    fold_id: int,
    mode: str,
) -> tuple[float, np.ndarray, np.ndarray, list[dict], dict]:
    if len(df_fold) == 0:
        return capital_in, np.array([], dtype=np.float64), np.array([], dtype=np.float64), [], {}

    global_to_local = dict(zip(df_fold["global_idx"].astype(int).tolist(), df_fold.index.tolist()))
    fold_global_last = int(df_fold["global_idx"].iloc[-1])

    capital = float(capital_in)
    equity_curve = []
    strategy_returns = []
    trade_log = []

    i = 0
    cooldown = 0

    while i < len(df_fold) - 1:
        row = df_fold.iloc[i]

        if cooldown > 0:
            cooldown -= 1
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            i += 1
            continue

        long_pass, long_info = evaluate_long_pass(row, mode)
        short_pass, short_info = evaluate_short_pass(row, mode)

        if not long_pass and not short_pass:
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            i += 1
            continue

        if long_pass and short_pass:
            rank_long = dual_conflict_rank("long", row)
            rank_short = dual_conflict_rank("short", row)
            if rank_long >= rank_short:
                short_pass = False
                chosen_rank = rank_long
            else:
                long_pass = False
                chosen_rank = rank_short
        else:
            chosen_rank = dual_conflict_rank("long", row) if long_pass else dual_conflict_rank("short", row)

        if long_pass:
            direction = "long"
            regime = str(row["regime_label"])
            params = get_long_regime_params(regime)
            prob = float(row["prob_long"])
            threshold = float(params["threshold"])
            setup_score = float(row["setup_score_long_regime"])
            ev_used = float(row["ev_long_regime"])
        else:
            direction = "short"
            regime = str(row["regime_label"])
            params = get_short_regime_params(regime)
            prob = float(row["prob_short"])
            threshold = float(params["threshold"])
            setup_score = float(row["setup_score_short_regime"])
            ev_used = float(row["ev_short_regime"])

        size = compute_side_size(
            direction=direction,
            prob=prob,
            threshold=threshold,
            setup_score=setup_score,
            ev_used=ev_used,
        )

        sim = simulate_trade_within_fold(
            df_fold=df_fold,
            entry_idx_local=i,
            direction=direction,
            fold_global_last=fold_global_last,
            global_to_local=global_to_local,
        )

        if sim is None:
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            i += 1
            continue

        gross_ret, exit_idx_local, outcome = sim

        fee_cost = cost_for_transition(0.0, size, CONFIG["fee_rate"], CONFIG["slippage_bps"])
        fee_cost += cost_for_transition(size, 0.0, CONFIG["fee_rate"], CONFIG["slippage_bps"])
        net_ret = size * gross_ret - fee_cost

        capital *= (1.0 + net_ret)

        trade_log.append(
            {
                "mode": mode,
                "fold_id": fold_id,
                "entry_idx_local": int(i),
                "exit_idx_local": int(exit_idx_local),
                "entry_global_idx": int(df_fold.iloc[i]["global_idx"]),
                "exit_global_idx": int(df_fold.iloc[exit_idx_local]["global_idx"]),
                "timestamp_entry": str(df_fold.iloc[i]["timestamp"]),
                "timestamp_exit": str(df_fold.iloc[exit_idx_local]["timestamp"]),
                "direction": direction,
                "regime_label": regime,
                "prob_long": float(row["prob_long"]),
                "prob_short": float(row["prob_short"]),
                "setup_score_long_regime": float(row["setup_score_long_regime"]),
                "setup_score_short_regime": float(row["setup_score_short_regime"]),
                "ev_long_regime": float(row["ev_long_regime"]),
                "ev_short_regime": float(row["ev_short_regime"]),
                "size": float(size),
                "gross_ret": float(gross_ret),
                "net_ret": float(net_ret),
                "outcome": outcome,
                "selected_long_regime": int(row["selected_long_regime"]),
                "selected_short_regime": int(row["selected_short_regime"]),
                "long_threshold_used": float(long_info.get("threshold", np.nan)) if isinstance(long_info, dict) else np.nan,
                "short_threshold_used": float(short_info.get("threshold", np.nan)) if isinstance(short_info, dict) else np.nan,
                "long_min_ev_used": float(long_info.get("min_ev", np.nan)) if isinstance(long_info, dict) else np.nan,
                "short_min_ev_used": float(short_info.get("min_ev", np.nan)) if isinstance(short_info, dict) else np.nan,
                "long_min_setup_used": float(long_info.get("min_setup", np.nan)) if isinstance(long_info, dict) else np.nan,
                "short_min_setup_used": float(short_info.get("min_setup", np.nan)) if isinstance(short_info, dict) else np.nan,
                "conflict_rank_used": float(chosen_rank),
            }
        )

        strategy_returns.append(net_ret)
        equity_curve.append(capital)
        i = max(i + 1, exit_idx_local + 1)
        cooldown = CONFIG["cooldown_bars"]

    equity_curve = np.array(equity_curve, dtype=np.float64)
    strategy_returns = np.array(strategy_returns, dtype=np.float64)
    trades_df = pd.DataFrame(trade_log)

    metrics = {
        "mode": mode,
        "fold_id": int(fold_id),
        "capital_in": float(capital_in),
        "capital_out": float(capital),
        "fold_return": float(capital / capital_in - 1.0) if capital_in > 0 else 0.0,
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "sortino_ratio": sortino_ratio(strategy_returns),
        "n_trades": int(len(trades_df)),
        "avg_trade_return": float(trades_df["net_ret"].mean()) if len(trades_df) else 0.0,
        "win_rate_trade": float((trades_df["net_ret"] > 0).mean()) if len(trades_df) else 0.0,
        "long_trades": int((trades_df["direction"] == "long").sum()) if len(trades_df) else 0,
        "short_trades": int((trades_df["direction"] == "short").sum()) if len(trades_df) else 0,
        "avg_size": float(trades_df["size"].mean()) if len(trades_df) else 0.0,
        "avg_ev_long": float(trades_df["ev_long_regime"].mean()) if len(trades_df) else 0.0,
        "avg_ev_short": float(trades_df["ev_short_regime"].mean()) if len(trades_df) else 0.0,
    }

    return capital, equity_curve, strategy_returns, trade_log, metrics


# =============================================================================
# ATTRIBUTION
# =============================================================================

def build_oos_attribution(trades_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if len(trades_df) == 0:
        empty = pd.DataFrame()
        return {
            "by_direction": empty,
            "by_regime": empty,
            "by_ev_bucket": empty,
            "by_score_bucket": empty,
        }

    t = trades_df.copy()

    t["ev_used"] = np.where(t["direction"] == "long", t["ev_long_regime"], t["ev_short_regime"])
    t["score_used"] = np.where(
        t["direction"] == "long",
        t["setup_score_long_regime"],
        t["setup_score_short_regime"],
    )

    t["ev_bucket"] = bucketize_series(
        t["ev_used"],
        bins=[-1.0, 0.004, 0.006, 0.008, 1.0],
        labels=["<=0.004", "(0.004,0.006]", "(0.006,0.008]", ">0.008"],
    )
    t["score_bucket"] = bucketize_series(
        t["score_used"],
        bins=[-1.0, 1.5, 2.0, 3.0, 99.0],
        labels=["<=1.5", "(1.5,2.0]", "(2.0,3.0]", ">3.0"],
    )

    def summarize(group_col: str) -> pd.DataFrame:
        g = t.groupby(group_col, dropna=False).agg(
            n_trades=("net_ret", "size"),
            total_net_ret=("net_ret", "sum"),
            avg_trade_ret=("net_ret", "mean"),
            win_rate=("net_ret", lambda x: float((x > 0).mean())),
            avg_size=("size", "mean"),
        ).reset_index()
        return g.sort_values("total_net_ret", ascending=False).reset_index(drop=True)

    return {
        "by_direction": summarize("direction"),
        "by_regime": summarize("regime_label"),
        "by_ev_bucket": summarize("ev_bucket"),
        "by_score_bucket": summarize("score_bucket"),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_seed(CONFIG["seed"])

    logger.info("=" * 100)
    logger.info("10 WALKFORWARD META MODEL V6 - REGIME AWARE")
    logger.info("=" * 100)
    logger.info(json.dumps(CONFIG, indent=2))
    logger.info(f"Device: {CONFIG['device']}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain 'timestamp'")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["global_idx"] = np.arange(len(df), dtype=np.int64)

    folds = build_folds(
        n_rows=len(df),
        train_bars=CONFIG["train_bars"],
        test_bars=CONFIG["test_bars"],
        step_bars=CONFIG["step_bars"],
        expanding_window=CONFIG["expanding_window"],
    )

    if len(folds) == 0:
        raise ValueError("No folds generated. Recheck train/test/step bars.")

    logger.info(f"Rows dataset: {len(df):,}")
    logger.info(f"Walk-forward folds: {len(folds)}")

    mode_state = {
        mode: {
            "capital": float(CONFIG["initial_capital"]),
            "fold_metrics": [],
            "trades": [],
            "equity_segments": [],
            "return_segments": [],
        }
        for mode in CONFIG["modes"]
    }

    predictions_dump = []

    for fold in folds:
        fold_id = fold["fold_id"]
        logger.info("-" * 100)
        logger.info(
            f"FOLD {fold_id} | "
            f"train=[{fold['train_start']}:{fold['train_end']}) | "
            f"test=[{fold['test_start']}:{fold['test_end']})"
        )
        logger.info("-" * 100)

        df_train = df.iloc[fold["train_start"]:fold["train_end"]].copy().reset_index(drop=True)
        df_test = df.iloc[fold["test_start"]:fold["test_end"]].copy().reset_index(drop=True)

        long_bundle = train_one_side_fold(
            df_train=df_train,
            candidate_col="candidate_long",
            label_col="tb_long_label",
            side_name=f"fold{fold_id}_long",
        )
        short_bundle = train_one_side_fold(
            df_train=df_train,
            candidate_col="candidate_short",
            label_col="tb_short_label",
            side_name=f"fold{fold_id}_short",
        )

        prob_long = predict_side_bundle(long_bundle, df_test)
        prob_short = predict_side_bundle(short_bundle, df_test)

        df_fold_bt = prepare_test_frame(df_test, prob_long=prob_long, prob_short=prob_short)

        predictions_dump.append(
            df_fold_bt[
                [
                    "timestamp",
                    "global_idx",
                    "candidate_long",
                    "candidate_short",
                    "prob_long",
                    "prob_short",
                    "ev_long_regime",
                    "ev_short_regime",
                    "selected_long_regime",
                    "selected_short_regime",
                    "setup_score_long_regime",
                    "setup_score_short_regime",
                    "regime_label",
                ]
            ].assign(fold_id=fold_id)
        )

        for mode in CONFIG["modes"]:
            capital_before = mode_state[mode]["capital"]

            capital_after, equity_curve, strategy_returns, trade_log, fold_metrics = run_backtest_on_fold(
                df_fold=df_fold_bt,
                capital_in=capital_before,
                fold_id=fold_id,
                mode=mode,
            )

            fold_metrics["train_rows"] = int(len(df_train))
            fold_metrics["test_rows"] = int(len(df_test))
            fold_metrics["long_train_stats"] = long_bundle.stats
            fold_metrics["short_train_stats"] = short_bundle.stats

            mode_state[mode]["capital"] = capital_after
            mode_state[mode]["fold_metrics"].append(fold_metrics)
            mode_state[mode]["trades"].extend(trade_log)

            if len(equity_curve) > 0:
                mode_state[mode]["equity_segments"].append(equity_curve)
            if len(strategy_returns) > 0:
                mode_state[mode]["return_segments"].append(strategy_returns)

            logger.info(
                f"FOLD {fold_id} | mode={mode} | "
                f"capital_in={capital_before:.2f} | capital_out={capital_after:.2f} | "
                f"n_trades={fold_metrics['n_trades']} | "
                f"fold_return={fold_metrics['fold_return']:.4%} | "
                f"avg_trade={fold_metrics['avg_trade_return']:.5f}"
            )

    predictions_df = pd.concat(predictions_dump, axis=0, ignore_index=True) if predictions_dump else pd.DataFrame()
    predictions_df.to_csv(OUT_DIR / "walkforward_meta_model_v6_predictions.csv", index=False)

    summary_rows = []

    for mode in CONFIG["modes"]:
        capital = mode_state[mode]["capital"]

        equity_all = (
            np.concatenate(mode_state[mode]["equity_segments"]).astype(np.float64)
            if len(mode_state[mode]["equity_segments"]) > 0 else np.array([], dtype=np.float64)
        )
        returns_all = (
            np.concatenate(mode_state[mode]["return_segments"]).astype(np.float64)
            if len(mode_state[mode]["return_segments"]) > 0 else np.array([], dtype=np.float64)
        )
        trades_df = pd.DataFrame(mode_state[mode]["trades"])
        folds_df = pd.DataFrame(
            [{k: v for k, v in fm.items() if not isinstance(v, dict)} for fm in mode_state[mode]["fold_metrics"]]
        )

        overall_metrics = {
            "mode": mode,
            "initial_capital": float(CONFIG["initial_capital"]),
            "final_capital": float(capital),
            "total_return": float(capital / CONFIG["initial_capital"] - 1.0),
            "max_drawdown": max_drawdown(equity_all),
            "sharpe_ratio": sharpe_ratio(returns_all),
            "sortino_ratio": sortino_ratio(returns_all),
            "n_folds": int(len(mode_state[mode]["fold_metrics"])),
            "n_trades": int(len(trades_df)),
            "avg_trade_return": float(trades_df["net_ret"].mean()) if len(trades_df) else 0.0,
            "win_rate_trade": float((trades_df["net_ret"] > 0).mean()) if len(trades_df) else 0.0,
            "long_trades": int((trades_df["direction"] == "long").sum()) if len(trades_df) else 0,
            "short_trades": int((trades_df["direction"] == "short").sum()) if len(trades_df) else 0,
            "avg_size": float(trades_df["size"].mean()) if len(trades_df) else 0.0,
            "avg_ev_long": float(trades_df["ev_long_regime"].mean()) if len(trades_df) else 0.0,
            "avg_ev_short": float(trades_df["ev_short_regime"].mean()) if len(trades_df) else 0.0,
        }

        attribution = build_oos_attribution(trades_df)

        mode_dir = OUT_DIR / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        if len(equity_all) > 0:
            pd.DataFrame(
                {
                    "step": np.arange(len(equity_all), dtype=np.int64),
                    "equity": equity_all,
                    "strategy_return": returns_all,
                }
            ).to_csv(mode_dir / "equity.csv", index=False)

        trades_df.to_csv(mode_dir / "trades.csv", index=False)
        folds_df.to_csv(mode_dir / "folds.csv", index=False)

        attribution["by_direction"].to_csv(mode_dir / "oos_by_direction.csv", index=False)
        attribution["by_regime"].to_csv(mode_dir / "oos_by_regime.csv", index=False)
        attribution["by_ev_bucket"].to_csv(mode_dir / "oos_by_ev_bucket.csv", index=False)
        attribution["by_score_bucket"].to_csv(mode_dir / "oos_by_score_bucket.csv", index=False)

        with open(mode_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, ensure_ascii=False, indent=2)

        with open(mode_dir / "fold_details.json", "w", encoding="utf-8") as f:
            json.dump(mode_state[mode]["fold_metrics"], f, ensure_ascii=False, indent=2)

        summary_rows.append(overall_metrics)

        logger.info("=" * 100)
        logger.info(f"MODE COMPLETE: {mode}")
        logger.info("=" * 100)
        for k, v in overall_metrics.items():
            logger.info(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_modes.csv", index=False)

    logger.info("=" * 100)
    logger.info("WALKFORWARD REGIME-AWARE COMPLETE")
    logger.info("=" * 100)
    logger.info(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
