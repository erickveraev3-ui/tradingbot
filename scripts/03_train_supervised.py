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
# CONFIG
# ============================================================

CONFIG = {
    "seq_len": 64,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    "batch_size": 256,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.15,
    "regime_classes": 4,

    "early_stopping_patience": 6,
    "seed": 42,
}

TARGET_COLUMNS = [
    "target_return_1h",
    "target_return_4h",
    "target_direction",
]

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
    """
    Régimen simple y estable para entrenamiento multitarea.
    0 = trend_up
    1 = trend_down
    2 = mean_reverting
    3 = neutral/high_noise
    """
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


def temporal_split_indices(n: int, train_ratio: float, val_ratio: float):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def fit_standard_scaler(train_array: np.ndarray):
    mean = np.nanmean(train_array, axis=0)
    std = np.nanstd(train_array, axis=0) + 1e-8
    return mean, std


def apply_standard_scaler(array: np.ndarray, mean: np.ndarray, std: np.ndarray):
    x = (array - mean) / std
    x = np.clip(x, -5, 5)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


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
# TRAIN / EVAL
# ============================================================

def make_loaders(df: pd.DataFrame, seq_len: int, batch_size: int):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "target_regime"]
    if not feature_cols:
        raise ValueError("No se encontraron columnas de features")

    logger.info(f"Features usadas: {len(feature_cols)}")
    logger.info(", ".join(feature_cols[:20]) + (" ..." if len(feature_cols) > 20 else ""))

    train_end, val_end = temporal_split_indices(len(df), CONFIG["train_ratio"], CONFIG["val_ratio"])

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    train_x_raw = df_train[feature_cols].values
    val_x_raw = df_val[feature_cols].values
    test_x_raw = df_test[feature_cols].values

    scaler_mean, scaler_std = fit_standard_scaler(train_x_raw)

    train_x = apply_standard_scaler(train_x_raw, scaler_mean, scaler_std)
    val_x = apply_standard_scaler(val_x_raw, scaler_mean, scaler_std)
    test_x = apply_standard_scaler(test_x_raw, scaler_mean, scaler_std)

    train_ret = df_train[["target_return_1h", "target_return_4h"]].values
    val_ret = df_val[["target_return_1h", "target_return_4h"]].values
    test_ret = df_test[["target_return_1h", "target_return_4h"]].values

    train_dir = df_train["target_direction"].values
    val_dir = df_val["target_direction"].values
    test_dir = df_test["target_direction"].values

    train_reg = df_train["target_regime"].values
    val_reg = df_val["target_regime"].values
    test_reg = df_test["target_regime"].values

    train_ds = SequenceDataset(train_x, train_ret, train_dir, train_reg, seq_len)
    val_ds = SequenceDataset(val_x, val_ret, val_dir, val_reg, seq_len)
    test_ds = SequenceDataset(test_x, test_ret, test_dir, test_reg, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    scaler_payload = {
        "mean": scaler_mean.tolist(),
        "std": scaler_std.tolist(),
        "feature_columns": feature_cols,
    }

    split_payload = {
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
    }

    return train_loader, val_loader, test_loader, scaler_payload, split_payload, len(feature_cols)


def evaluate(model, loader, device, ret_loss_fn, cls_loss_fn):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    dir_correct = 0
    dir_total = 0

    reg_correct = 0
    reg_total = 0

    ret_preds_all = []
    ret_true_all = []

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

            ret_preds_all.append(out["return_preds"].cpu().numpy())
            ret_true_all.append(y_ret.cpu().numpy())

    ret_preds_all = np.vstack(ret_preds_all)
    ret_true_all = np.vstack(ret_true_all)

    ret_mae = np.mean(np.abs(ret_preds_all - ret_true_all), axis=0)

    return {
        "loss": total_loss / max(total_batches, 1),
        "direction_acc": dir_correct / max(dir_total, 1),
        "regime_acc": reg_correct / max(reg_total, 1),
        "ret1h_mae": float(ret_mae[0]),
        "ret4h_mae": float(ret_mae[1]),
    }


def main():
    set_seed(CONFIG["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO SUPERVISADO MULTITAREA")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")

    data_path = root_dir / "data" / "processed" / "dataset_btc_context_1h.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"No existe {data_path}. Ejecuta antes scripts/02_build_dataset.py"
        )

    df = pd.read_csv(data_path)
    if "timestamp" not in df.columns:
        raise ValueError("El dataset no contiene columna timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["target_regime"] = infer_regime_labels(df)

    logger.info(f"Filas dataset: {len(df):,}")
    logger.info(f"Rango: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    train_loader, val_loader, test_loader, scaler_payload, split_payload, input_dim = make_loaders(
        df=df,
        seq_len=CONFIG["seq_len"],
        batch_size=CONFIG["batch_size"],
    )

    model_config = GRUAlphaConfig(
        input_dim=input_dim,
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        regime_classes=CONFIG["regime_classes"],
    )

    model = GRUAlphaModel(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    ret_loss_fn = nn.SmoothL1Loss()
    cls_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience = 0
    history = []

    for epoch in range(1, CONFIG["epochs"] + 1):
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

        train_loss = running_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device, ret_loss_fn, cls_loss_fn)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"dir_acc={val_metrics['direction_acc']:.3f} | "
            f"reg_acc={val_metrics['regime_acc']:.3f} | "
            f"ret1h_mae={val_metrics['ret1h_mae']:.6f} | "
            f"ret4h_mae={val_metrics['ret4h_mae']:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience = 0

            model_path = root_dir / "artifacts" / "models" / "gru_alpha_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "train_config": CONFIG,
                },
                model_path,
            )
            logger.info(f"✅ Nuevo mejor modelo guardado en {model_path}")
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                logger.info("⏹ Early stopping activado")
                break

    scaler_path = root_dir / "artifacts" / "scalers" / "gru_alpha_scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, ensure_ascii=False, indent=2)

    history_path = root_dir / "artifacts" / "reports" / "train_history_supervised.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    split_path = root_dir / "artifacts" / "reports" / "dataset_split_summary.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)

    # Cargar mejor modelo para test final
    best_ckpt = torch.load(root_dir / "artifacts" / "models" / "gru_alpha_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, ret_loss_fn, cls_loss_fn)

    test_path = root_dir / "artifacts" / "reports" / "test_metrics_supervised.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("TEST FINAL")
    logger.info("=" * 70)
    for k, v in test_metrics.items():
        logger.info(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    logger.info("Entrenamiento completado.")


if __name__ == "__main__":
    main()
