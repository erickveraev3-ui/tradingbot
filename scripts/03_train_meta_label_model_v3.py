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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger


DATA_PATH = root_dir / "data/processed/dataset_btc_triple_barrier_1h.csv"
MODEL_DIR = root_dir / "artifacts/models"
SCALER_DIR = root_dir / "artifacts/scalers"
REPORT_DIR = root_dir / "artifacts/reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "batch_size": 256,
    "epochs": 40,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "dropout": 0.25,
    "early_stopping_patience": 6,
    "seed": 42,
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TradingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


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


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Selección defensiva:
    excluye cualquier columna raw/meta/label que pueda filtrar futuro.
    """
    feature_cols = []

    for c in df.columns:
        if c == "timestamp":
            continue

        # OHLCV bruto lo excluimos para evitar que domine sobre señales derivadas
        if c in ["open", "high", "low", "close", "volume", "quote_volume", "trades"]:
            continue

        # Todas las columnas triple-barrier fuera
        if c.startswith("tb_"):
            continue

        # Eventos futuros fuera
        if c.startswith("event_"):
            continue
        if c.startswith("future_"):
            continue
        if c.startswith("target_"):
            continue

        # candidatos fuera: el filtro ya se aplica antes
        if c in ["candidate_long", "candidate_short"]:
            continue

        feature_cols.append(c)

    return feature_cols


def temporal_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def train_one_side(
    df: pd.DataFrame,
    candidate_col: str,
    label_col: str,
    model_name: str,
):
    logger.info("=" * 80)
    logger.info(f"Training {model_name}")
    logger.info("=" * 80)

    df_side = df[df[candidate_col] == 1].copy().reset_index(drop=True)
    if df_side.empty:
        raise ValueError(f"{model_name}: no hay filas con {candidate_col} == 1")

    # label binario: solo wins
    y = (df_side[label_col] == 1).astype(np.float32).values

    feature_cols = select_feature_columns(df_side)
    X = df_side[feature_cols].values.astype(np.float32)

    logger.info(f"{model_name} rows: {len(df_side):,}")
    logger.info(f"{model_name} features: {len(feature_cols)}")
    logger.info(f"{model_name} positive rate total: {y.mean():.4f}")

    X_train, X_val, y_train, y_val = temporal_split(X, y, train_ratio=0.8)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = neg_count / max(pos_count, 1.0)

    logger.info(f"{model_name} positive rate train: {y_train.mean():.4f}")
    logger.info(f"{model_name} pos_weight: {pos_weight:.2f}")

    train_ds = TradingDataset(X_train, y_train)
    val_ds = TradingDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    model = MetaModel(input_dim=X.shape[1], dropout=CONFIG["dropout"]).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    best_ap = -np.inf
    best_state = None
    patience = 0
    history = []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).unsqueeze(1)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy().flatten()

                preds.extend(prob.tolist())
                trues.extend(yb.numpy().tolist())

        preds = np.array(preds, dtype=np.float32)
        trues = np.array(trues, dtype=np.float32)

        auc = safe_auc(trues, preds)
        ap = safe_ap(trues, preds)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": auc,
            "val_ap": ap,
        })

        logger.info(
            f"{model_name} | epoch {epoch:02d} | "
            f"loss {train_loss:.4f} | auc {auc:.4f} | ap {ap:.4f}"
        )

        metric_for_es = ap if not np.isnan(ap) else -np.inf

        if metric_for_es > best_ap:
            best_ap = metric_for_es
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_cols": feature_cols,
            }
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                logger.info(f"{model_name} early stopping")
                break

    if best_state is None:
        raise RuntimeError(f"{model_name}: no se guardó mejor estado")

    # guardar modelo
    model_path = MODEL_DIR / f"{model_name}.pt"
    torch.save(best_state["model_state_dict"], model_path)

    # guardar scaler + columnas
    scaler_payload = {
        "feature_columns": feature_cols,
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    scaler_path = SCALER_DIR / f"{model_name}_scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, ensure_ascii=False, indent=2)

    # guardar history
    history_path = REPORT_DIR / f"{model_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    logger.info(f"Saved {model_name} -> {model_path}")
    logger.info(f"Saved scaler -> {scaler_path}")
    logger.info(f"Saved history -> {history_path}")

    return {
        "model_name": model_name,
        "rows": len(df_side),
        "positive_rate_total": float(y.mean()),
        "positive_rate_train": float(y_train.mean()),
        "best_val_ap": float(best_ap),
    }


def main():
    set_seed(CONFIG["seed"])
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Loading dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Rows total: {len(df):,}")
    logger.info(f"Columns total: {len(df.columns)}")

    long_stats = train_one_side(
        df=df,
        candidate_col="candidate_long",
        label_col="tb_long_label",
        model_name="meta_model_long_v3",
    )

    short_stats = train_one_side(
        df=df,
        candidate_col="candidate_short",
        label_col="tb_short_label",
        model_name="meta_model_short_v3",
    )

    summary = {
        "long": long_stats,
        "short": short_stats,
    }

    summary_path = REPORT_DIR / "meta_model_v3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
