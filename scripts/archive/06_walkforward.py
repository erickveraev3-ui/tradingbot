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

from src.models.gru_alpha_model import GRUAlphaModel, GRUAlphaConfig


# ============================================================
# WALK-FORWARD CONFIG
# ============================================================

TRAIN_CONFIG = {
    "seq_len": 64,
    "batch_size": 256,
    "epochs": 18,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.15,
    "regime_classes": 4,
    "early_stopping_patience": 4,
    "seed": 42,
}

# Campeón provisional del sweep
BT_CONFIG = {
    "seq_len": 64,
    "initial_capital": 10000.0,
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,
    "threshold_long": 0.0040,
    "threshold_short": -0.0040,
    "cooldown_bars": 3,
    "min_hold_bars": 2,
    "max_position": 1.0,
    "use_short": False,
    "min_direction_confidence": 0.54,
    "min_edge_over_cost": 0.0010,
    "allowed_regimes_long": [0, 2],
    "allowed_regimes_short": [1],
}

# Ventanas en proporción del dataset total
WF_CONFIG = {
    "train_window_ratio": 0.40,
    "test_window_ratio": 0.10,
    "step_ratio": 0.10,
}

NON_FEATURE_COLUMNS = set([
    "timestamp",
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades",
    "target_return_1h",
    "target_return_4h",
    "target_direction",
])


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_regime_labels(df: pd.DataFrame) -> pd.Series:
    ret_24 = df["return_24"] if "return_24" in df.columns else df["close"].pct_change(24)
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


def fit_standard_scaler(train_array: np.ndarray):
    mean = np.nanmean(train_array, axis=0)
    std = np.nanstd(train_array, axis=0) + 1e-8
    return mean, std


def apply_standard_scaler(array: np.ndarray, mean: np.ndarray, std: np.ndarray):
    x = (array - mean) / std
    x = np.clip(x, -5, 5)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


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


def cost_for_transition(pos_prev: int, pos_new: int, fee_rate: float, slippage_bps: float) -> float:
    traded = abs(pos_new - pos_prev)
    if traded == 0:
        return 0.0
    cost = traded * fee_rate
    cost += traded * (slippage_bps / 10000.0)
    return cost


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target_returns: np.ndarray,
        target_direction: np.ndarray,
        target_regime: np.ndarray,
        seq_len: int,
    ):
        self.features = features.astype(np.float32)
        self.target_returns = target_returns.astype(np.float32)
        self.target_direction = target_direction.astype(np.int64)
        self.target_regime = target_regime.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        end = idx + self.seq_len
        x = self.features[idx:end]
        y_ret = self.target_returns[end - 1]
        y_dir = self.target_direction[end - 1]
        y_reg = self.target_regime[end - 1]

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y_ret": torch.tensor(y_ret, dtype=torch.float32),
            "y_dir": torch.tensor(y_dir, dtype=torch.long),
            "y_reg": torch.tensor(y_reg, dtype=torch.long),
        }


# ============================================================
# TRAINING
# ============================================================

def prepare_arrays(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "target_regime"]
    x = df[feature_cols].values
    y_ret = df[["target_return_1h", "target_return_4h"]].values
    y_dir = df["target_direction"].values
    y_reg = df["target_regime"].values
    return feature_cols, x, y_ret, y_dir, y_reg


def evaluate(model, loader, device, ret_loss_fn, cls_loss_fn):
    model.eval()

    total_loss = 0.0
    total_batches = 0
    dir_correct = 0
    dir_total = 0
    reg_correct = 0
    reg_total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_ret = batch["y_ret"].to(device)
            y_dir = batch["y_dir"].to(device)
            y_reg = batch["y_reg"].to(device)

            out = model(x)

            loss_ret = ret_loss_fn(out["return_preds"], y_ret)
            loss_dir = cls_loss_fn(out["direction_logits"], y_dir)
            loss_reg = cls_loss_fn(out["regime_logits"], y_reg)

            loss = loss_ret + 0.5 * loss_dir + 0.3 * loss_reg

            total_loss += loss.item()
            total_batches += 1

            dir_pred = torch.argmax(out["direction_logits"], dim=-1)
            reg_pred = torch.argmax(out["regime_logits"], dim=-1)

            dir_correct += (dir_pred == y_dir).sum().item()
            dir_total += y_dir.numel()

            reg_correct += (reg_pred == y_reg).sum().item()
            reg_total += y_reg.numel()

    return {
        "loss": total_loss / max(total_batches, 1),
        "direction_acc": dir_correct / max(dir_total, 1),
        "regime_acc": reg_correct / max(reg_total, 1),
    }


def train_one_fold(df_train: pd.DataFrame, df_test: pd.DataFrame, device: str):
    feature_cols, x_train_raw, y_train_ret, y_train_dir, y_train_reg = prepare_arrays(df_train)
    _, x_test_raw, y_test_ret, y_test_dir, y_test_reg = prepare_arrays(df_test)

    scaler_mean, scaler_std = fit_standard_scaler(x_train_raw)

    x_train = apply_standard_scaler(x_train_raw, scaler_mean, scaler_std)
    x_test = apply_standard_scaler(x_test_raw, scaler_mean, scaler_std)

    # train/val split dentro del fold de train
    split_idx = int(len(x_train) * 0.85)

    x_tr, x_val = x_train[:split_idx], x_train[split_idx:]
    y_tr_ret, y_val_ret = y_train_ret[:split_idx], y_train_ret[split_idx:]
    y_tr_dir, y_val_dir = y_train_dir[:split_idx], y_train_dir[split_idx:]
    y_tr_reg, y_val_reg = y_train_reg[:split_idx], y_train_reg[split_idx:]

    train_ds = SequenceDataset(x_tr, y_tr_ret, y_tr_dir, y_tr_reg, TRAIN_CONFIG["seq_len"])
    val_ds = SequenceDataset(x_val, y_val_ret, y_val_dir, y_val_reg, TRAIN_CONFIG["seq_len"])
    test_ds = SequenceDataset(x_test, y_test_ret, y_test_dir, y_test_reg, TRAIN_CONFIG["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

    model_config = GRUAlphaConfig(
        input_dim=len(feature_cols),
        hidden_dim=TRAIN_CONFIG["hidden_dim"],
        num_layers=TRAIN_CONFIG["num_layers"],
        dropout=TRAIN_CONFIG["dropout"],
        regime_classes=TRAIN_CONFIG["regime_classes"],
    )
    model = GRUAlphaModel(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    ret_loss_fn = nn.SmoothL1Loss()
    cls_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y_ret = batch["y_ret"].to(device)
            y_dir = batch["y_dir"].to(device)
            y_reg = batch["y_reg"].to(device)

            optimizer.zero_grad()
            out = model(x)

            loss_ret = ret_loss_fn(out["return_preds"], y_ret)
            loss_dir = cls_loss_fn(out["direction_logits"], y_dir)
            loss_reg = cls_loss_fn(out["regime_logits"], y_reg)
            loss = loss_ret + 0.5 * loss_dir + 0.3 * loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate(model, val_loader, device, ret_loss_fn, cls_loss_fn)
        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_config),
                "feature_cols": feature_cols,
                "scaler_mean": scaler_mean.tolist(),
                "scaler_std": scaler_std.tolist(),
            }
            patience = 0
        else:
            patience += 1
            if patience >= TRAIN_CONFIG["early_stopping_patience"]:
                break

    if best_state is None:
        raise RuntimeError("No se guardó mejor estado en el fold")

    model.load_state_dict(best_state["model_state_dict"])
    return model, feature_cols, scaler_mean, scaler_std, test_loader, x_test


# ============================================================
# BACKTEST DEL FOLD
# ============================================================

def run_backtest(df_test: pd.DataFrame, features_scaled: np.ndarray, model: GRUAlphaModel, seq_len: int, device: str):
    capital = BT_CONFIG["initial_capital"]
    position = 0
    cooldown = 0
    hold_bars = 0

    strategy_returns = []
    equity_curve = []
    n_trades = 0
    long_trades = 0
    short_trades = 0
    flat_transitions = 0
    total_cost = 0.0
    diag_positions = []

    model.eval()

    for idx in range(seq_len - 1, len(df_test) - 1):
        x_seq = features_scaled[idx - seq_len + 1: idx + 1]
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model.predict(x_tensor)

        pred_ret_1h = float(out["return_preds"][0, 0].cpu().item())
        pred_ret_4h = float(out["return_preds"][0, 1].cpu().item())
        dir_prob = out["direction_prob"][0].cpu().numpy()
        reg_prob = out["regime_prob"][0].cpu().numpy()

        p_short = float(dir_prob[0])
        p_long = float(dir_prob[1])
        regime = int(np.argmax(reg_prob))

        close_now = float(df_test.iloc[idx]["close"])
        close_next = float(df_test.iloc[idx + 1]["close"])

        score = 0.70 * pred_ret_1h + 0.30 * pred_ret_4h
        target_position = position

        est_entry_cost = cost_for_transition(position, 1 if score > 0 else -1, BT_CONFIG["fee_rate"], BT_CONFIG["slippage_bps"])
        est_edge_long = score - est_entry_cost
        est_edge_short = -score - est_entry_cost

        long_signal = (
            score >= BT_CONFIG["threshold_long"]
            and p_long >= BT_CONFIG["min_direction_confidence"]
            and est_edge_long >= BT_CONFIG["min_edge_over_cost"]
            and regime in BT_CONFIG["allowed_regimes_long"]
        )

        short_signal = (
            BT_CONFIG["use_short"]
            and score <= BT_CONFIG["threshold_short"]
            and p_short >= BT_CONFIG["min_direction_confidence"]
            and est_edge_short >= BT_CONFIG["min_edge_over_cost"]
            and regime in BT_CONFIG["allowed_regimes_short"]
        )

        flat_signal = not long_signal and not short_signal

        if cooldown > 0:
            cooldown -= 1

        if position != 0:
            hold_bars += 1
        else:
            hold_bars = 0

        if cooldown > 0:
            target_position = position
        elif position != 0 and hold_bars < BT_CONFIG["min_hold_bars"]:
            target_position = position
        else:
            if position == 0:
                if long_signal:
                    target_position = 1
                elif short_signal:
                    target_position = -1
                else:
                    target_position = 0
            elif position == 1:
                if long_signal:
                    target_position = 1
                elif flat_signal:
                    target_position = 0
                elif short_signal:
                    target_position = 0
            elif position == -1:
                if short_signal:
                    target_position = -1
                elif flat_signal:
                    target_position = 0
                elif long_signal:
                    target_position = 0

        fee_cost = cost_for_transition(position, target_position, BT_CONFIG["fee_rate"], BT_CONFIG["slippage_bps"])

        if target_position != position:
            n_trades += 1
            total_cost += fee_cost
            cooldown = BT_CONFIG["cooldown_bars"]

            if target_position == 1:
                long_trades += 1
            elif target_position == -1:
                short_trades += 1
            elif target_position == 0:
                flat_transitions += 1
                hold_bars = 0

        raw_ret = (close_next / close_now) - 1.0
        strat_ret = target_position * raw_ret - fee_cost
        capital *= (1.0 + strat_ret)

        position = target_position
        strategy_returns.append(strat_ret)
        equity_curve.append(capital)
        diag_positions.append(position)

    strategy_returns = np.array(strategy_returns, dtype=np.float64)
    equity_curve = np.array(equity_curve, dtype=np.float64)
    diag_positions = np.array(diag_positions, dtype=np.int64)

    return {
        "final_capital": float(equity_curve[-1]) if len(equity_curve) else BT_CONFIG["initial_capital"],
        "total_return": float(equity_curve[-1] / BT_CONFIG["initial_capital"] - 1.0) if len(equity_curve) else 0.0,
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "sortino_ratio": sortino_ratio(strategy_returns),
        "n_trades": int(n_trades),
        "avg_trade_cost": float(total_cost / n_trades) if n_trades > 0 else 0.0,
        "long_trades": int(long_trades),
        "short_trades": int(short_trades),
        "flat_transitions": int(flat_transitions),
        "pct_time_in_market": float(np.mean(diag_positions != 0)) if len(diag_positions) else 0.0,
        "pct_long": float(np.mean(diag_positions == 1)) if len(diag_positions) else 0.0,
        "pct_short": float(np.mean(diag_positions == -1)) if len(diag_positions) else 0.0,
        "mean_bar_return": float(strategy_returns.mean()) if len(strategy_returns) else 0.0,
        "std_bar_return": float(strategy_returns.std()) if len(strategy_returns) else 0.0,
    }


# ============================================================
# WALK-FORWARD
# ============================================================

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
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")

    data_path = root_dir / "data" / "processed" / "dataset_btc_context_1h.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"No existe {data_path}. Ejecuta antes scripts/02_build_dataset.py")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_regime"] = infer_regime_labels(df)

    splits = generate_walkforward_splits(len(df))
    if not splits:
        raise ValueError("No se pudieron generar folds walk-forward")

    logger.info(f"Total folds: {len(splits)}")

    all_results = []

    for split in splits:
        fold = split["fold"]
        logger.info("-" * 70)
        logger.info(f"Fold {fold}")

        df_train = df.iloc[split["train_start"]:split["train_end"]].copy().reset_index(drop=True)
        df_test = df.iloc[split["test_start"]:split["test_end"]].copy().reset_index(drop=True)

        logger.info(
            f"Train: {df_train['timestamp'].min()} -> {df_train['timestamp'].max()} | "
            f"{len(df_train):,} filas"
        )
        logger.info(
            f"Test:  {df_test['timestamp'].min()} -> {df_test['timestamp'].max()} | "
            f"{len(df_test):,} filas"
        )

        if len(df_train) < TRAIN_CONFIG["seq_len"] + 200 or len(df_test) < TRAIN_CONFIG["seq_len"] + 50:
            logger.warning(f"Fold {fold} omitido por tamaño insuficiente")
            continue

        model, feature_cols, scaler_mean, scaler_std, _, _ = train_one_fold(df_train, df_test, device)
        x_test = apply_standard_scaler(df_test[feature_cols].values, scaler_mean, scaler_std)

        bt_metrics = run_backtest(
            df_test=df_test,
            features_scaled=x_test,
            model=model,
            seq_len=TRAIN_CONFIG["seq_len"],
            device=device,
        )

        row = {
            "fold": fold,
            "train_start_ts": str(df_train["timestamp"].min()),
            "train_end_ts": str(df_train["timestamp"].max()),
            "test_start_ts": str(df_test["timestamp"].min()),
            "test_end_ts": str(df_test["timestamp"].max()),
            "train_rows": len(df_train),
            "test_rows": len(df_test),
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
        raise RuntimeError("No se generaron resultados en walk-forward")

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
        "worst_fold_return": float(results_df["total_return"].min()),
        "best_fold_return": float(results_df["total_return"].max()),
    }

    reports_dir = root_dir / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    folds_path = reports_dir / "walkforward_results.csv"
    summary_path = reports_dir / "walkforward_summary.json"

    results_df.to_csv(folds_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("WALK-FORWARD SUMMARY")
    logger.info("=" * 70)
    for k, v in summary.items():
        logger.info(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    logger.info(f"Resultados por fold: {folds_path}")
    logger.info(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
