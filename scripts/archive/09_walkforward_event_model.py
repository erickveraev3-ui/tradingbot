from __future__ import annotations

import sys
import json
from pathlib import Path
from dataclasses import asdict

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.models.gru_event_model import GRUEventModel, GRUEventConfig


EVENT_TARGETS = [
    "target_event_long",
    "target_event_short",
    "event_breakout_up",
    "event_breakdown_down",
]

RAW_EXCLUDE = {
    "timestamp",
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades",
}

TRAIN_CONFIG = {
    "seq_len": 96,
    "batch_size": 256,
    "epochs": 18,
    "lr": 8e-4,
    "weight_decay": 1e-5,
    "hidden_dim": 160,
    "num_layers": 2,
    "dropout": 0.20,
    "n_regime_classes": 4,
    "early_stopping_patience": 4,
    "seed": 42,
    "clip_grad_norm": 1.0,
    "calibration_steps": 300,
    "calibration_lr": 0.03,
}

BT_CONFIG = {
    "seq_len": 96,
    "initial_capital": 10000.0,
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,
    "use_short": True,
    "cooldown_bars": 3,
    "min_hold_bars": 3,
    "allowed_regimes_long": [0, 2],
    "allowed_regimes_short": [1, 2],
    "min_vol_ratio": 0.60,
    "max_vol_ratio": 1.90,
    "max_abs_bb_position": 1.90,
    "min_long_short_gap": 0.05,
    "min_short_long_gap": 0.05,
    "min_breakout_bonus": 0.55,
    "min_breakdown_bonus": 0.55,
    "base_position": 0.20,
    "max_position": 1.00,
    "signal_scale": 1.80,
    "breakout_scale": 0.40,
    "breakdown_scale": 0.40,
    "regime_mult_long": {0: 1.00, 1: 0.00, 2: 0.65, 3: 0.00},
    "regime_mult_short": {0: 0.00, 1: 1.00, 2: 0.55, 3: 0.00},
}

WF_CONFIG = {
    "train_window_ratio": 0.40,
    "test_window_ratio": 0.10,
    "step_ratio": 0.10,
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_regime_labels(df: pd.DataFrame) -> pd.Series:
    ret_24 = df["ret_24"] if "ret_24" in df.columns else df["close"].pct_change(24)
    adx = df["adx"] if "adx" in df.columns else pd.Series(np.zeros(len(df)))
    vol_ratio = df["vol_ratio"] if "vol_ratio" in df.columns else pd.Series(np.ones(len(df)))
    bb_pos = df["bb_position"] if "bb_position" in df.columns else pd.Series(np.zeros(len(df)))

    regime = np.full(len(df), 3, dtype=np.int64)
    trend_up = (ret_24 > 0) & (adx > 0.20)
    trend_down = (ret_24 < 0) & (adx > 0.20)
    mean_rev = (adx < 0.12) & (bb_pos.abs() > 0.8) & (vol_ratio < 1.2)

    regime[trend_up.fillna(False).values] = 0
    regime[trend_down.fillna(False).values] = 1
    regime[mean_rev.fillna(False).values] = 2
    return pd.Series(regime, index=df.index, name="target_regime")


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = []
    for c in df.columns:
        if c in RAW_EXCLUDE:
            continue
        if c == "target_regime":
            continue
        if c.startswith("target_"):
            continue
        if c.startswith("event_"):
            continue
        if c.startswith("future_"):
            continue
        feature_cols.append(c)
    return feature_cols


def fit_robust_scaler(train_array: np.ndarray):
    median = np.nanmedian(train_array, axis=0)
    mad = np.nanmedian(np.abs(train_array - median), axis=0)
    scale = mad * 1.4826 + 1e-8
    return median, scale


def apply_robust_scaler(array: np.ndarray, median: np.ndarray, scale: np.ndarray):
    x = (array - median) / scale
    x = np.clip(x, -8, 8)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if y_true.sum() == 0:
            return float("nan")
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def compute_pos_weight(y_train_events: np.ndarray) -> np.ndarray:
    pos = y_train_events.sum(axis=0)
    neg = len(y_train_events) - pos
    pos_weight = (neg + 1e-8) / (pos + 1e-8)
    pos_weight = np.clip(pos_weight, 1.0, 50.0)
    return pos_weight.astype(np.float32)


class EventSequenceDataset(Dataset):
    def __init__(self, features, y_events, y_regime, seq_len):
        self.features = features.astype(np.float32)
        self.y_events = y_events.astype(np.float32)
        self.y_regime = y_regime.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        end = idx + self.seq_len
        return {
            "x": torch.tensor(self.features[idx:end], dtype=torch.float32),
            "y_evt": torch.tensor(self.y_events[end - 1], dtype=torch.float32),
            "y_reg": torch.tensor(self.y_regime[end - 1], dtype=torch.long),
        }


class MultiLabelTemperatureScaler(nn.Module):
    def __init__(self, n_outputs: int):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(n_outputs))
        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)
        return (logits + self.bias) / temp


def fit_calibrator(val_logits: np.ndarray, val_targets: np.ndarray, device: str):
    calibrator = MultiLabelTemperatureScaler(val_logits.shape[1]).to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=TRAIN_CONFIG["calibration_lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.tensor(val_logits, dtype=torch.float32, device=device)
    y = torch.tensor(val_targets, dtype=torch.float32, device=device)

    calibrator.train()
    for _ in range(TRAIN_CONFIG["calibration_steps"]):
        optimizer.zero_grad()
        logits_cal = calibrator(x)
        loss = loss_fn(logits_cal, y)
        loss.backward()
        optimizer.step()

    payload = {
        "log_temp": calibrator.log_temp.detach().cpu().numpy().tolist(),
        "bias": calibrator.bias.detach().cpu().numpy().tolist(),
    }
    return payload


def apply_calibration_np(logits: np.ndarray, calibration_payload: dict) -> np.ndarray:
    log_temp = np.array(calibration_payload["log_temp"], dtype=np.float32)
    bias = np.array(calibration_payload["bias"], dtype=np.float32)
    temp = np.exp(log_temp).clip(1e-3, 100.0)
    logits_cal = (logits + bias) / temp
    probs = 1.0 / (1.0 + np.exp(-logits_cal))
    return probs


def fbeta_score_np(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-8)


def tune_thresholds(val_probs: np.ndarray, val_targets: np.ndarray):
    thresholds = {}
    grid = np.arange(0.35, 0.85 + 1e-9, 0.025)
    head_min_precision = {
        "target_event_long": 0.55,
        "target_event_short": 0.55,
        "event_breakout_up": 0.40,
        "event_breakdown_down": 0.40,
    }

    for i, head in enumerate(EVENT_TARGETS):
        y_true = val_targets[:, i]
        y_prob = val_probs[:, i]

        best = {"threshold": 0.5, "fbeta": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        min_prec = head_min_precision[head]

        for thr in grid:
            y_pred = (y_prob >= thr).astype(int)
            if y_pred.sum() == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

            fbeta = fbeta_score_np(precision, recall, beta=0.5)
            if precision < min_prec:
                fbeta *= 0.5

            if fbeta > best["fbeta"]:
                best = {
                    "threshold": float(thr),
                    "fbeta": float(fbeta),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }

        thresholds[head] = best

    return thresholds


def collect_outputs(model, loader, device):
    model.eval()
    all_logits, all_evt, all_reg = [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_evt = batch["y_evt"].cpu().numpy()
            y_reg = batch["y_reg"].cpu().numpy()

            out = model(x)
            all_logits.append(out["event_logits"].cpu().numpy())
            all_evt.append(y_evt)
            all_reg.append(y_reg)

    return {
        "event_logits": np.vstack(all_logits),
        "event_targets": np.vstack(all_evt),
        "regime_targets": np.concatenate(all_reg),
    }


def evaluate_macro_ap(model, loader, device):
    outputs = collect_outputs(model, loader, device)
    probs = 1.0 / (1.0 + np.exp(-outputs["event_logits"]))
    aps = []
    for i in range(len(EVENT_TARGETS)):
        ap = safe_ap(outputs["event_targets"][:, i], probs[:, i])
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else float("nan")


def train_one_fold(df_train: pd.DataFrame, device: str):
    feature_cols = select_feature_columns(df_train)

    # split train/val interno
    split_idx = int(len(df_train) * 0.85)
    fold_train = df_train.iloc[:split_idx].copy()
    fold_val = df_train.iloc[split_idx:].copy()

    x_train_raw = fold_train[feature_cols].values
    x_val_raw = fold_val[feature_cols].values

    scaler_median, scaler_scale = fit_robust_scaler(x_train_raw)

    x_train = apply_robust_scaler(x_train_raw, scaler_median, scaler_scale)
    x_val = apply_robust_scaler(x_val_raw, scaler_median, scaler_scale)

    y_train_evt = fold_train[EVENT_TARGETS].values.astype(np.float32)
    y_val_evt = fold_val[EVENT_TARGETS].values.astype(np.float32)

    y_train_reg = fold_train["target_regime"].values.astype(np.int64)
    y_val_reg = fold_val["target_regime"].values.astype(np.int64)

    train_ds = EventSequenceDataset(x_train, y_train_evt, y_train_reg, TRAIN_CONFIG["seq_len"])
    val_ds = EventSequenceDataset(x_val, y_val_evt, y_val_reg, TRAIN_CONFIG["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, drop_last=False)

    model_config = GRUEventConfig(
        input_dim=len(feature_cols),
        hidden_dim=TRAIN_CONFIG["hidden_dim"],
        num_layers=TRAIN_CONFIG["num_layers"],
        dropout=TRAIN_CONFIG["dropout"],
        n_event_outputs=len(EVENT_TARGETS),
        n_regime_classes=TRAIN_CONFIG["n_regime_classes"],
    )

    model = GRUEventModel(model_config).to(device)

    pos_weight = compute_pos_weight(y_train_evt)
    event_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    regime_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    use_amp = device == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_ap = -np.inf
    best_state = None
    patience = 0

    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        model.train()

        for batch in train_loader:
            x = batch["x"].to(device)
            y_evt = batch["y_evt"].to(device)
            y_reg = batch["y_reg"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x)
                loss_evt = event_loss_fn(out["event_logits"], y_evt)
                loss_reg = regime_loss_fn(out["regime_logits"], y_reg)
                loss = loss_evt + 0.25 * loss_reg

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAIN_CONFIG["clip_grad_norm"])
            scaler_amp.step(optimizer)
            scaler_amp.update()

        val_ap = evaluate_macro_ap(model, val_loader, device)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_state = {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_config),
                "feature_cols": feature_cols,
                "scaler_median": scaler_median.tolist(),
                "scaler_scale": scaler_scale.tolist(),
            }
            patience = 0
        else:
            patience += 1
            if patience >= TRAIN_CONFIG["early_stopping_patience"]:
                break

    if best_state is None:
        raise RuntimeError("No se obtuvo mejor modelo en el fold")

    model.load_state_dict(best_state["model_state_dict"])

    val_outputs = collect_outputs(model, val_loader, device)
    val_probs_uncal = 1.0 / (1.0 + np.exp(-val_outputs["event_logits"]))
    calibration_payload = fit_calibrator(val_outputs["event_logits"], val_outputs["event_targets"], device)
    val_probs_cal = apply_calibration_np(val_outputs["event_logits"], calibration_payload)
    thresholds = tune_thresholds(val_probs_cal, val_outputs["event_targets"])

    return {
        "model": model,
        "feature_cols": feature_cols,
        "scaler_median": np.array(best_state["scaler_median"], dtype=np.float32),
        "scaler_scale": np.array(best_state["scaler_scale"], dtype=np.float32),
        "calibration_payload": calibration_payload,
        "thresholds": thresholds,
        "val_macro_ap_uncal": float(np.nanmean([
            safe_ap(val_outputs["event_targets"][:, i], val_probs_uncal[:, i]) for i in range(len(EVENT_TARGETS))
        ])),
        "val_macro_ap_cal": float(np.nanmean([
            safe_ap(val_outputs["event_targets"][:, i], val_probs_cal[:, i]) for i in range(len(EVENT_TARGETS))
        ])),
    }


def cost_for_transition(pos_prev: float, pos_new: float, fee_rate: float, slippage_bps: float) -> float:
    turnover = abs(pos_new - pos_prev)
    if turnover == 0:
        return 0.0
    cost = turnover * fee_rate
    cost += turnover * (slippage_bps / 10000.0)
    return cost


def compute_position_size(direction: int, p_dir: float, p_opp: float, p_break: float, regime: int) -> float:
    if direction == 0:
        return 0.0

    gap = max(0.0, p_dir - p_opp)
    size = BT_CONFIG["base_position"]
    size += BT_CONFIG["signal_scale"] * gap

    if direction > 0 and p_break >= BT_CONFIG["min_breakout_bonus"]:
        size += BT_CONFIG["breakout_scale"] * (p_break - BT_CONFIG["min_breakout_bonus"])
    if direction < 0 and p_break >= BT_CONFIG["min_breakdown_bonus"]:
        size += BT_CONFIG["breakdown_scale"] * (p_break - BT_CONFIG["min_breakdown_bonus"])

    if direction > 0:
        size *= BT_CONFIG["regime_mult_long"].get(regime, 0.0)
    else:
        size *= BT_CONFIG["regime_mult_short"].get(regime, 0.0)

    size = min(size, BT_CONFIG["max_position"])
    size = max(0.0, size)
    return float(direction) * size


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
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


def run_backtest_fold(df_test: pd.DataFrame, artifacts: dict, device: str):
    feature_cols = artifacts["feature_cols"]
    median = artifacts["scaler_median"]
    scale = artifacts["scaler_scale"]
    model = artifacts["model"]
    calibration_payload = artifacts["calibration_payload"]
    thresholds = artifacts["thresholds"]

    x_test = apply_robust_scaler(df_test[feature_cols].values, median, scale)

    capital = BT_CONFIG["initial_capital"]
    position = 0.0
    cooldown = 0
    hold_bars = 0

    strategy_returns = []
    equity_curve = []
    n_trades = 0
    total_cost = 0.0
    positions = []

    long_thr = thresholds["target_event_long"]["threshold"]
    short_thr = thresholds["target_event_short"]["threshold"]
    breakout_thr = thresholds["event_breakout_up"]["threshold"]
    breakdown_thr = thresholds["event_breakdown_down"]["threshold"]

    model.eval()

    for idx in range(BT_CONFIG["seq_len"] - 1, len(df_test) - 1):
        row = df_test.iloc[idx]
        close_now = float(row["close"])
        close_next = float(df_test.iloc[idx + 1]["close"])
        regime = int(row["target_regime"])
        vol_ratio = float(row["vol_ratio"]) if "vol_ratio" in df_test.columns and pd.notna(row["vol_ratio"]) else 1.0
        bb_position = float(row["bb_position"]) if "bb_position" in df_test.columns and pd.notna(row["bb_position"]) else 0.0

        x_seq = x_test[idx - BT_CONFIG["seq_len"] + 1: idx + 1]
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(x_tensor)

        probs = apply_calibration_np(out["event_logits"].cpu().numpy(), calibration_payload)[0]

        p_long = float(probs[0])
        p_short = float(probs[1])
        p_breakout = float(probs[2])
        p_breakdown = float(probs[3])

        vol_filter_ok = (
            BT_CONFIG["min_vol_ratio"] <= vol_ratio <= BT_CONFIG["max_vol_ratio"]
            and abs(bb_position) <= BT_CONFIG["max_abs_bb_position"]
        )

        long_signal = (
            vol_filter_ok
            and regime in BT_CONFIG["allowed_regimes_long"]
            and p_long >= long_thr
            and (p_long - p_short) >= BT_CONFIG["min_long_short_gap"]
        )
        short_signal = (
            BT_CONFIG["use_short"]
            and vol_filter_ok
            and regime in BT_CONFIG["allowed_regimes_short"]
            and p_short >= short_thr
            and (p_short - p_long) >= BT_CONFIG["min_short_long_gap"]
        )

        flat_signal = not long_signal and not short_signal

        if cooldown > 0:
            cooldown -= 1

        if abs(position) > 1e-12:
            hold_bars += 1
        else:
            hold_bars = 0

        target_position = position

        if cooldown > 0:
            target_position = position
        elif abs(position) > 1e-12 and hold_bars < BT_CONFIG["min_hold_bars"]:
            target_position = position
        else:
            if abs(position) < 1e-12:
                if long_signal:
                    target_position = compute_position_size(1, p_long, p_short, p_breakout, regime)
                elif short_signal:
                    target_position = compute_position_size(-1, p_short, p_long, p_breakdown, regime)
                else:
                    target_position = 0.0
            elif position > 0:
                if long_signal:
                    target_position = compute_position_size(1, p_long, p_short, p_breakout, regime)
                elif flat_signal or short_signal:
                    target_position = 0.0
            elif position < 0:
                if short_signal:
                    target_position = compute_position_size(-1, p_short, p_long, p_breakdown, regime)
                elif flat_signal or long_signal:
                    target_position = 0.0

        fee_cost = cost_for_transition(position, target_position, BT_CONFIG["fee_rate"], BT_CONFIG["slippage_bps"])

        if abs(target_position - position) > 1e-12:
            n_trades += 1
            total_cost += fee_cost
            cooldown = BT_CONFIG["cooldown_bars"]
            if abs(target_position) < 1e-12:
                hold_bars = 0

        raw_ret = (close_next / close_now) - 1.0
        strat_ret = target_position * raw_ret - fee_cost
        capital *= (1.0 + strat_ret)

        position = target_position
        strategy_returns.append(strat_ret)
        equity_curve.append(capital)
        positions.append(position)

    strategy_returns = np.array(strategy_returns, dtype=np.float64)
    equity_curve = np.array(equity_curve, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)

    return {
        "final_capital": float(equity_curve[-1]) if len(equity_curve) else BT_CONFIG["initial_capital"],
        "total_return": float(equity_curve[-1] / BT_CONFIG["initial_capital"] - 1.0) if len(equity_curve) else 0.0,
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "sortino_ratio": sortino_ratio(strategy_returns),
        "n_trades": int(n_trades),
        "avg_trade_cost": float(total_cost / n_trades) if n_trades > 0 else 0.0,
        "pct_time_in_market": float(np.mean(np.abs(positions) > 1e-12)) if len(positions) else 0.0,
        "avg_abs_position": float(np.mean(np.abs(positions))) if len(positions) else 0.0,
        "mean_bar_return": float(strategy_returns.mean()) if len(strategy_returns) else 0.0,
        "std_bar_return": float(strategy_returns.std()) if len(strategy_returns) else 0.0,
    }


def generate_walkforward_splits(n: int):
    train_window = int(n * WF_CONFIG["train_window_ratio"])
    test_window = int(n * WF_CONFIG["test_window_ratio"])
    step = int(n * WF_CONFIG["step_ratio"])

    splits = []
    start = 0
    fold = 0

    while True:
        train_start = start
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window

        if test_end > n:
            break

        splits.append({
            "fold": fold,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        start += step
        fold += 1

    return splits


def main():
    set_seed(TRAIN_CONFIG["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 80)
    logger.info("WALK-FORWARD DEL MODELO DE EVENTOS")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")

    data_path = root_dir / "data" / "processed" / "dataset_btc_event_1h.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"No existe {data_path}")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_regime"] = infer_regime_labels(df)

    splits = generate_walkforward_splits(len(df))
    if not splits:
        raise ValueError("No se pudieron generar folds")

    logger.info(f"Total folds walk-forward: {len(splits)}")

    all_results = []

    for split in splits:
        fold = split["fold"]
        logger.info("-" * 80)
        logger.info(f"Fold {fold}")

        df_train = df.iloc[split["train_start"]:split["train_end"]].copy().reset_index(drop=True)
        df_test = df.iloc[split["test_start"]:split["test_end"]].copy().reset_index(drop=True)

        logger.info(
            f"Train: {df_train['timestamp'].min()} -> {df_train['timestamp'].max()} | {len(df_train):,} filas"
        )
        logger.info(
            f"Test : {df_test['timestamp'].min()} -> {df_test['timestamp'].max()} | {len(df_test):,} filas"
        )

        if len(df_train) < TRAIN_CONFIG["seq_len"] + 500 or len(df_test) < TRAIN_CONFIG["seq_len"] + 100:
            logger.warning(f"Fold {fold} omitido por tamaño insuficiente")
            continue

        artifacts = train_one_fold(df_train, device)
        bt_metrics = run_backtest_fold(df_test, artifacts, device)

        row = {
            "fold": fold,
            "train_start_ts": str(df_train["timestamp"].min()),
            "train_end_ts": str(df_train["timestamp"].max()),
            "test_start_ts": str(df_test["timestamp"].min()),
            "test_end_ts": str(df_test["timestamp"].max()),
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "val_macro_ap_uncal": artifacts["val_macro_ap_uncal"],
            "val_macro_ap_cal": artifacts["val_macro_ap_cal"],
            **bt_metrics,
        }
        all_results.append(row)

        logger.info(
            f"Fold {fold} | "
            f"ret={bt_metrics['total_return']:.4f} | "
            f"dd={bt_metrics['max_drawdown']:.4f} | "
            f"sh={bt_metrics['sharpe_ratio']:.3f} | "
            f"trades={bt_metrics['n_trades']} | "
            f"expo={bt_metrics['pct_time_in_market']:.3f}"
        )

    if not all_results:
        raise RuntimeError("No hubo resultados walk-forward")

    results_df = pd.DataFrame(all_results)

    summary = {
        "n_folds": int(len(results_df)),
        "mean_total_return": float(results_df["total_return"].mean()),
        "median_total_return": float(results_df["total_return"].median()),
        "mean_max_drawdown": float(results_df["max_drawdown"].mean()),
        "median_max_drawdown": float(results_df["max_drawdown"].median()),
        "mean_sharpe_ratio": float(results_df["sharpe_ratio"].mean()),
        "median_sharpe_ratio": float(results_df["sharpe_ratio"].median()),
        "positive_return_folds_pct": float((results_df["total_return"] > 0).mean()),
        "positive_sharpe_folds_pct": float((results_df["sharpe_ratio"] > 0).mean()),
        "mean_n_trades": float(results_df["n_trades"].mean()),
        "mean_pct_time_in_market": float(results_df["pct_time_in_market"].mean()),
        "mean_avg_abs_position": float(results_df["avg_abs_position"].mean()),
        "mean_val_macro_ap_cal": float(results_df["val_macro_ap_cal"].mean()),
        "worst_fold_return": float(results_df["total_return"].min()),
        "best_fold_return": float(results_df["total_return"].max()),
    }

    reports_dir = root_dir / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    folds_path = reports_dir / "walkforward_event_results.csv"
    summary_path = reports_dir / "walkforward_event_summary.json"

    results_df.to_csv(folds_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info("WALK-FORWARD EVENT SUMMARY")
    logger.info("=" * 80)
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.6f}")
        else:
            logger.info(f"{k}: {v}")

    logger.info(f"Resultados por fold: {folds_path}")
    logger.info(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
