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
    brier_score_loss,
)

from src.models.gru_event_model import GRUEventModel, GRUEventConfig


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "seq_len": 96,                     # más contexto para estructura/impulso
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    "batch_size": 256,
    "epochs": 30,
    "lr": 8e-4,
    "weight_decay": 1e-5,
    "hidden_dim": 160,
    "num_layers": 2,
    "dropout": 0.20,
    "n_regime_classes": 4,

    "early_stopping_patience": 6,
    "seed": 42,
    "clip_grad_norm": 1.0,

    # para calibration
    "calibration_steps": 400,
    "calibration_lr": 0.03,

    # thresholds a buscar
    "threshold_grid_start": 0.35,
    "threshold_grid_end": 0.85,
    "threshold_grid_step": 0.025,
}

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


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_regime_labels(df: pd.DataFrame) -> pd.Series:
    """
    Tarea auxiliar: ayudar al modelo a situarse en el contexto de mercado.
    """
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


def temporal_split_indices(n: int, train_ratio: float, val_ratio: float):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Selección defensiva de features:
    - excluye raw OHLCV
    - excluye cualquier columna target/event/future que pueda fugar información
    """
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
    """
    BCE pos_weight por cabeza.
    """
    pos = y_train_events.sum(axis=0)
    neg = len(y_train_events) - pos
    pos_weight = (neg + 1e-8) / (pos + 1e-8)
    pos_weight = np.clip(pos_weight, 1.0, 50.0)
    return pos_weight.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class EventSequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        y_events: np.ndarray,
        y_regime: np.ndarray,
        seq_len: int,
    ):
        self.features = features.astype(np.float32)
        self.y_events = y_events.astype(np.float32)
        self.y_regime = y_regime.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        end = idx + self.seq_len
        x = self.features[idx:end]
        y_evt = self.y_events[end - 1]
        y_reg = self.y_regime[end - 1]

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y_evt": torch.tensor(y_evt, dtype=torch.float32),
            "y_reg": torch.tensor(y_reg, dtype=torch.long),
        }


# ============================================================
# CALIBRATION
# ============================================================

class MultiLabelTemperatureScaler(nn.Module):
    """
    Calibration ligera pero útil:
    logits_cal = (logits + bias) / temperature
    por cabeza.
    """
    def __init__(self, n_outputs: int):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(n_outputs))
        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)
        return (logits + self.bias) / temp


def fit_calibrator(val_logits: np.ndarray, val_targets: np.ndarray, device: str):
    calibrator = MultiLabelTemperatureScaler(val_logits.shape[1]).to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=CONFIG["calibration_lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.tensor(val_logits, dtype=torch.float32, device=device)
    y = torch.tensor(val_targets, dtype=torch.float32, device=device)

    calibrator.train()
    for _ in range(CONFIG["calibration_steps"]):
        optimizer.zero_grad()
        logits_cal = calibrator(x)
        loss = loss_fn(logits_cal, y)
        loss.backward()
        optimizer.step()

    calibrator.eval()
    with torch.no_grad():
        logits_cal = calibrator(x)
        probs_cal = torch.sigmoid(logits_cal).cpu().numpy()

    payload = {
        "log_temp": calibrator.log_temp.detach().cpu().numpy().tolist(),
        "bias": calibrator.bias.detach().cpu().numpy().tolist(),
    }
    return calibrator, probs_cal, payload


def apply_calibration_np(logits: np.ndarray, calibration_payload: dict) -> np.ndarray:
    log_temp = np.array(calibration_payload["log_temp"], dtype=np.float32)
    bias = np.array(calibration_payload["bias"], dtype=np.float32)
    temp = np.exp(log_temp).clip(1e-3, 100.0)
    logits_cal = (logits + bias) / temp
    probs = 1.0 / (1.0 + np.exp(-logits_cal))
    return probs


# ============================================================
# THRESHOLD TUNING
# ============================================================

def fbeta_score_np(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-8)


def tune_thresholds(val_probs: np.ndarray, val_targets: np.ndarray):
    thresholds = {}
    grid = np.arange(
        CONFIG["threshold_grid_start"],
        CONFIG["threshold_grid_end"] + 1e-9,
        CONFIG["threshold_grid_step"],
    )

    # Queremos favorecer precision frente a recall
    head_min_precision = {
        "target_event_long": 0.55,
        "target_event_short": 0.55,
        "event_breakout_up": 0.40,
        "event_breakdown_down": 0.40,
    }

    for i, head in enumerate(EVENT_TARGETS):
        y_true = val_targets[:, i]
        y_prob = val_probs[:, i]

        best = {
            "threshold": 0.5,
            "fbeta": -1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

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

            # penalizar thresholds con precision demasiado baja
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


# ============================================================
# EVALUATION
# ============================================================

def collect_outputs(model, loader, device):
    model.eval()

    all_logits = []
    all_regime_logits = []
    all_evt = []
    all_reg = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_evt = batch["y_evt"].cpu().numpy()
            y_reg = batch["y_reg"].cpu().numpy()

            out = model(x)

            all_logits.append(out["event_logits"].cpu().numpy())
            all_regime_logits.append(out["regime_logits"].cpu().numpy())
            all_evt.append(y_evt)
            all_reg.append(y_reg)

    return {
        "event_logits": np.vstack(all_logits),
        "regime_logits": np.vstack(all_regime_logits),
        "event_targets": np.vstack(all_evt),
        "regime_targets": np.concatenate(all_reg),
    }


def evaluate_multitask(
    model,
    loader,
    device,
    event_loss_fn,
    regime_loss_fn,
    calibration_payload=None,
    thresholds=None,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0
    regime_correct = 0
    regime_total = 0

    outputs = collect_outputs(model, loader, device)
    logits = outputs["event_logits"]
    regime_logits = outputs["regime_logits"]
    y_evt = outputs["event_targets"]
    y_reg = outputs["regime_targets"]

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            yy_evt = batch["y_evt"].to(device)
            yy_reg = batch["y_reg"].to(device)

            out = model(x)
            loss_evt = event_loss_fn(out["event_logits"], yy_evt)
            loss_reg = regime_loss_fn(out["regime_logits"], yy_reg)
            loss = loss_evt + 0.25 * loss_reg

            total_loss += loss.item()
            total_batches += 1

            reg_pred = torch.argmax(out["regime_logits"], dim=-1)
            regime_correct += (reg_pred == yy_reg).sum().item()
            regime_total += yy_reg.numel()

    if calibration_payload is None:
        probs = 1.0 / (1.0 + np.exp(-logits))
    else:
        probs = apply_calibration_np(logits, calibration_payload)

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "regime_acc": regime_correct / max(regime_total, 1),
    }

    ap_list = []
    auc_list = []
    brier_list = []

    for i, head in enumerate(EVENT_TARGETS):
        y_true = y_evt[:, i]
        y_prob = probs[:, i]

        ap = safe_ap(y_true, y_prob)
        auc = safe_auc(y_true, y_prob)

        metrics[f"{head}_ap"] = ap
        metrics[f"{head}_auc"] = auc

        if not np.isnan(ap):
            ap_list.append(ap)
        if not np.isnan(auc):
            auc_list.append(auc)

        try:
            metrics[f"{head}_brier"] = float(brier_score_loss(y_true, y_prob))
            brier_list.append(metrics[f"{head}_brier"])
        except Exception:
            metrics[f"{head}_brier"] = float("nan")

        if thresholds is not None and head in thresholds:
            thr = thresholds[head]["threshold"]
            y_pred = (y_prob >= thr).astype(int)

            metrics[f"{head}_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics[f"{head}_recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics[f"{head}_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            metrics[f"{head}_pred_rate"] = float(y_pred.mean())

    metrics["macro_ap"] = float(np.mean(ap_list)) if ap_list else float("nan")
    metrics["macro_auc"] = float(np.mean(auc_list)) if auc_list else float("nan")
    metrics["macro_brier"] = float(np.mean(brier_list)) if brier_list else float("nan")

    return metrics, outputs


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(CONFIG["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 80)
    logger.info("ENTRENAMIENTO DEL MODELO DE EVENTOS")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")

    data_path = root_dir / "data" / "processed" / "dataset_btc_event_1h.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"No existe {data_path}. Ejecuta antes scripts/02_build_event_dataset.py"
        )

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_regime"] = infer_regime_labels(df)

    feature_cols = select_feature_columns(df)
    logger.info(f"Filas dataset: {len(df):,}")
    logger.info(f"Features usadas: {len(feature_cols)}")
    logger.info(f"Rango temporal: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # Split temporal limpio
    train_end, val_end = temporal_split_indices(len(df), CONFIG["train_ratio"], CONFIG["val_ratio"])

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    logger.info(f"Train rows: {len(df_train):,}")
    logger.info(f"Val rows:   {len(df_val):,}")
    logger.info(f"Test rows:  {len(df_test):,}")

    x_train_raw = df_train[feature_cols].values
    x_val_raw = df_val[feature_cols].values
    x_test_raw = df_test[feature_cols].values

    # Robust scaling solo con train
    scaler_median, scaler_scale = fit_robust_scaler(x_train_raw)
    x_train = apply_robust_scaler(x_train_raw, scaler_median, scaler_scale)
    x_val = apply_robust_scaler(x_val_raw, scaler_median, scaler_scale)
    x_test = apply_robust_scaler(x_test_raw, scaler_median, scaler_scale)

    y_train_evt = df_train[EVENT_TARGETS].values.astype(np.float32)
    y_val_evt = df_val[EVENT_TARGETS].values.astype(np.float32)
    y_test_evt = df_test[EVENT_TARGETS].values.astype(np.float32)

    y_train_reg = df_train["target_regime"].values.astype(np.int64)
    y_val_reg = df_val["target_regime"].values.astype(np.int64)
    y_test_reg = df_test["target_regime"].values.astype(np.int64)

    train_ds = EventSequenceDataset(x_train, y_train_evt, y_train_reg, CONFIG["seq_len"])
    val_ds = EventSequenceDataset(x_val, y_val_evt, y_val_reg, CONFIG["seq_len"])
    test_ds = EventSequenceDataset(x_test, y_test_evt, y_test_reg, CONFIG["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=False)

    model_config = GRUEventConfig(
        input_dim=len(feature_cols),
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        n_event_outputs=len(EVENT_TARGETS),
        n_regime_classes=CONFIG["n_regime_classes"],
    )

    model = GRUEventModel(model_config).to(device)

    pos_weight = compute_pos_weight(y_train_evt)
    event_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    regime_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    use_amp = device == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_ap = -np.inf
    patience = 0
    history = []
    best_state = None

    logger.info(f"Pos weights eventos: {dict(zip(EVENT_TARGETS, pos_weight.tolist()))}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["clip_grad_norm"])
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_metrics, _ = evaluate_multitask(
            model=model,
            loader=val_loader,
            device=device,
            event_loss_fn=event_loss_fn,
            regime_loss_fn=regime_loss_fn,
            calibration_payload=None,
            thresholds=None,
        )

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
            f"val_macro_ap={val_metrics['macro_ap']:.4f} | "
            f"val_macro_auc={val_metrics['macro_auc']:.4f} | "
            f"val_regime_acc={val_metrics['regime_acc']:.4f}"
        )

        # criterio principal: macro AP en validación
        current_val_ap = val_metrics["macro_ap"] if not np.isnan(val_metrics["macro_ap"]) else -np.inf

        if current_val_ap > best_val_ap:
            best_val_ap = current_val_ap
            patience = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_config),
                "train_config": CONFIG,
            }
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                logger.info("⏹ Early stopping activado")
                break

    if best_state is None:
        raise RuntimeError("No se guardó ningún mejor modelo")

    # restaurar mejor modelo
    model.load_state_dict(best_state["model_state_dict"])

    # Calibration sobre validación
    val_metrics_uncal, val_outputs = evaluate_multitask(
        model=model,
        loader=val_loader,
        device=device,
        event_loss_fn=event_loss_fn,
        regime_loss_fn=regime_loss_fn,
        calibration_payload=None,
        thresholds=None,
    )

    calibrator, val_probs_cal, calibration_payload = fit_calibrator(
        val_logits=val_outputs["event_logits"],
        val_targets=val_outputs["event_targets"],
        device=device,
    )

    thresholds = tune_thresholds(
        val_probs=val_probs_cal,
        val_targets=val_outputs["event_targets"],
    )

    # Evaluación final en test con calibración + thresholds
    test_metrics, test_outputs = evaluate_multitask(
        model=model,
        loader=test_loader,
        device=device,
        event_loss_fn=event_loss_fn,
        regime_loss_fn=regime_loss_fn,
        calibration_payload=calibration_payload,
        thresholds=thresholds,
    )

    # Guardados
    models_dir = root_dir / "artifacts" / "models"
    scalers_dir = root_dir / "artifacts" / "scalers"
    reports_dir = root_dir / "artifacts" / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "gru_event_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": CONFIG,
            "event_targets": EVENT_TARGETS,
        },
        model_path,
    )

    scaler_path = scalers_dir / "gru_event_scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "type": "robust",
                "median": scaler_median.tolist(),
                "scale": scaler_scale.tolist(),
                "feature_columns": feature_cols,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    calibration_path = models_dir / "gru_event_calibration.json"
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration_payload, f, ensure_ascii=False, indent=2)

    thresholds_path = models_dir / "gru_event_thresholds.json"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    history_path = reports_dir / "train_history_event_model.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    test_metrics_path = reports_dir / "test_metrics_event_model.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # también guardamos probs calibradas del test para análisis/backtest posterior
    test_probs_cal = apply_calibration_np(test_outputs["event_logits"], calibration_payload)
    probs_df = pd.DataFrame(test_probs_cal, columns=[f"prob_{x}" for x in EVENT_TARGETS])
    probs_df["timestamp"] = df_test.iloc[CONFIG["seq_len"] - 1:].reset_index(drop=True)["timestamp"].astype(str)
    probs_df.to_csv(reports_dir / "test_event_probabilities.csv", index=False)

    logger.info("=" * 80)
    logger.info("RESULTADOS TEST - MODELO DE EVENTOS")
    logger.info("=" * 80)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.6f}")
        else:
            logger.info(f"{k}: {v}")

    logger.info("Thresholds seleccionados:")
    for head, payload in thresholds.items():
        logger.info(f"{head}: {payload}")

    logger.info(f"Modelo guardado en: {model_path}")
    logger.info(f"Scaler guardado en: {scaler_path}")
    logger.info(f"Calibration guardada en: {calibration_path}")
    logger.info(f"Thresholds guardados en: {thresholds_path}")
    logger.info(f"Historial guardado en: {history_path}")
    logger.info(f"Métricas test guardadas en: {test_metrics_path}")


if __name__ == "__main__":
    main()
