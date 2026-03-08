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


DATA_PATH = root_dir / "data/processed/dataset_btc_triple_barrier_1h.csv"
MODEL_DIR = root_dir / "artifacts/models"
SCALER_DIR = root_dir / "artifacts/scalers"
REPORT_DIR = root_dir / "artifacts/reports"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = {
    "initial_capital": 10000.0,
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,

    # Decisión
    "use_long": True,
    "use_short": True,
    "threshold_long": 0.62,
    "threshold_short": 0.62,
    "prob_margin": 0.06,

    # Ejecución
    "cooldown_bars": 2,

    # Sizing
    "base_size": 0.25,
    "max_size": 1.00,
    "size_scale": 1.50,
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


def cost_for_transition(pos_prev: float, pos_new: float, fee_rate: float, slippage_bps: float) -> float:
    turnover = abs(pos_new - pos_prev)
    if turnover == 0:
        return 0.0
    return turnover * fee_rate + turnover * (slippage_bps / 10000.0)


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


def compute_size(prob: float, threshold: float) -> float:
    excess = max(0.0, prob - threshold)
    size = CONFIG["base_size"] + CONFIG["size_scale"] * excess
    size = min(size, CONFIG["max_size"])
    size = max(0.0, size)
    return size


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
) -> tuple[float, int, str]:
    """
    Simula el trade usando las mismas columnas triple-barrier ya construidas.
    Devuelve:
    - retorno bruto del trade
    - exit_idx
    - outcome
    """
    if direction == "long":
        label = int(df.iloc[entry_idx]["tb_long_label"])
        ret = float(df.iloc[entry_idx]["tb_long_return"])
        hit_bar = int(df.iloc[entry_idx]["tb_long_hit_bar"])
    else:
        label = int(df.iloc[entry_idx]["tb_short_label"])
        ret = float(df.iloc[entry_idx]["tb_short_return"])
        hit_bar = int(df.iloc[entry_idx]["tb_short_hit_bar"])

    if hit_bar < entry_idx:
        hit_bar = entry_idx + 1

    outcome = "expiry"
    if label == 1:
        outcome = "tp"
    elif label == -1:
        outcome = "sl"

    return ret, hit_bar, outcome


def main():
    logger.info("=" * 80)
    logger.info("BACKTEST META MODEL V3")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe {DATA_PATH}")

    long_model_path = MODEL_DIR / "meta_model_long_v3.pt"
    short_model_path = MODEL_DIR / "meta_model_short_v3.pt"
    long_scaler_path = SCALER_DIR / "meta_model_long_v3_scaler.json"
    short_scaler_path = SCALER_DIR / "meta_model_short_v3_scaler.json"

    for p in [long_model_path, short_model_path, long_scaler_path, short_scaler_path]:
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
        p_long = torch.sigmoid(long_model(torch.tensor(X_long, dtype=torch.float32, device=DEVICE))).cpu().numpy().flatten()
        p_short = torch.sigmoid(short_model(torch.tensor(X_short, dtype=torch.float32, device=DEVICE))).cpu().numpy().flatten()

    capital = CONFIG["initial_capital"]
    equity_curve = []
    trade_log = []
    strategy_returns = []

    i = 0
    cooldown = 0

    while i < len(df_test) - 1:
        row = df_test.iloc[i]
        ts = row["timestamp"]

        if cooldown > 0:
            cooldown -= 1
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            i += 1
            continue

        cand_long = int(row["candidate_long"])
        cand_short = int(row["candidate_short"])

        prob_long = float(p_long[i])
        prob_short = float(p_short[i])

        take_long = (
            CONFIG["use_long"]
            and cand_long == 1
            and prob_long >= CONFIG["threshold_long"]
            and (prob_long - prob_short) >= CONFIG["prob_margin"]
        )

        take_short = (
            CONFIG["use_short"]
            and cand_short == 1
            and prob_short >= CONFIG["threshold_short"]
            and (prob_short - prob_long) >= CONFIG["prob_margin"]
        )

        if not take_long and not take_short:
            equity_curve.append(capital)
            strategy_returns.append(0.0)
            i += 1
            continue

        if take_long and take_short:
            # priorizar mayor confianza relativa
            if prob_long - prob_short >= prob_short - prob_long:
                take_short = False
            else:
                take_long = False

        if take_long:
            direction = "long"
            prob = prob_long
            size = compute_size(prob, CONFIG["threshold_long"])
        else:
            direction = "short"
            prob = prob_short
            size = compute_size(prob, CONFIG["threshold_short"])

        gross_ret, exit_idx, outcome = simulate_trade(df_test, i, direction)
        fee_cost = cost_for_transition(0.0, size, CONFIG["fee_rate"], CONFIG["slippage_bps"])
        fee_cost += cost_for_transition(size, 0.0, CONFIG["fee_rate"], CONFIG["slippage_bps"])

        net_ret = size * gross_ret - fee_cost
        capital *= (1.0 + net_ret)

        trade_log.append({
            "entry_idx": i,
            "exit_idx": exit_idx,
            "timestamp_entry": str(df_test.iloc[i]["timestamp"]),
            "timestamp_exit": str(df_test.iloc[min(exit_idx, len(df_test)-1)]["timestamp"]),
            "direction": direction,
            "prob_long": prob_long,
            "prob_short": prob_short,
            "size": size,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "outcome": outcome,
        })

        strategy_returns.append(net_ret)
        equity_curve.append(capital)

        # saltamos hasta la salida del trade
        i = max(i + 1, exit_idx + 1)
        cooldown = CONFIG["cooldown_bars"]

    equity_curve = np.array(equity_curve, dtype=np.float64)
    strategy_returns = np.array(strategy_returns, dtype=np.float64)
    trades_df = pd.DataFrame(trade_log)

    metrics = {
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
        "avg_prob_long": float(trades_df["prob_long"].mean()) if len(trades_df) else 0.0,
        "avg_prob_short": float(trades_df["prob_short"].mean()) if len(trades_df) else 0.0,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "step": np.arange(len(equity_curve)),
        "equity": equity_curve,
        "strategy_return": strategy_returns,
    }).to_csv(REPORT_DIR / "backtest_meta_model_v3_equity.csv", index=False)

    trades_df.to_csv(REPORT_DIR / "backtest_meta_model_v3_trades.csv", index=False)

    with open(REPORT_DIR / "backtest_meta_model_v3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info("RESULTADOS BACKTEST META MODEL V3")
    logger.info("=" * 80)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
