from __future__ import annotations

import sys
import json
import itertools
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.models.gru_alpha_model import GRUAlphaModel, GRUAlphaConfig


BASE_CONFIG = {
    "seq_len": 64,
    "initial_capital": 10000.0,
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,
    "threshold_long": 0.0035,
    "threshold_short": -0.0040,
    "cooldown_bars": 6,
    "min_hold_bars": 4,
    "max_position": 1.0,
    "use_short": False,
    "min_direction_confidence": 0.58,
    "min_edge_over_cost": 0.0012,
    "allowed_regimes_long": [0, 2],
    "allowed_regimes_short": [1],
}

# Grid razonable: no demasiado grande para no perder tiempo
SWEEP_GRID = {
    "threshold_long": [0.0025, 0.0030, 0.0035, 0.0040],
    "cooldown_bars": [3, 6, 9],
    "min_hold_bars": [2, 4, 6],
    "min_direction_confidence": [0.54, 0.56, 0.58],
    "min_edge_over_cost": [0.0008, 0.0010, 0.0012],
}


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


def apply_scaler(df: pd.DataFrame, feature_cols: list[str], scaler_payload: dict) -> np.ndarray:
    x = df[feature_cols].values.astype(np.float32)
    mean = np.array(scaler_payload["mean"], dtype=np.float32)
    std = np.array(scaler_payload["std"], dtype=np.float32) + 1e-8
    x = (x - mean) / std
    x = np.clip(x, -5, 5)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def build_test_split(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[val_end:].copy().reset_index(drop=True)


def max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(dd.min()) if len(dd) else 0.0


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 24 * 365) -> float:
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    if sigma < 1e-12:
        return 0.0
    return mu / sigma * np.sqrt(periods_per_year)


def cost_for_transition(pos_prev: int, pos_new: int, fee_rate: float, slippage_bps: float) -> float:
    traded = abs(pos_new - pos_prev)
    if traded == 0:
        return 0.0
    cost = traded * fee_rate
    cost += traded * (slippage_bps / 10000.0)
    return cost


def run_backtest(df: pd.DataFrame, features_scaled: np.ndarray, model: GRUAlphaModel, seq_len: int, device: str, cfg: dict):
    capital = cfg["initial_capital"]
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

    for idx in range(seq_len - 1, len(df) - 1):
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

        close_now = float(df.iloc[idx]["close"])
        close_next = float(df.iloc[idx + 1]["close"])

        score = 0.70 * pred_ret_1h + 0.30 * pred_ret_4h

        target_position = position

        est_entry_cost = cost_for_transition(position, 1 if score > 0 else -1, cfg["fee_rate"], cfg["slippage_bps"])
        est_edge_long = score - est_entry_cost
        est_edge_short = -score - est_entry_cost

        long_signal = (
            score >= cfg["threshold_long"]
            and p_long >= cfg["min_direction_confidence"]
            and est_edge_long >= cfg["min_edge_over_cost"]
            and regime in cfg["allowed_regimes_long"]
        )

        short_signal = (
            cfg["use_short"]
            and score <= cfg["threshold_short"]
            and p_short >= cfg["min_direction_confidence"]
            and est_edge_short >= cfg["min_edge_over_cost"]
            and regime in cfg["allowed_regimes_short"]
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

        elif position != 0 and hold_bars < cfg["min_hold_bars"]:
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
                    target_position = 0  # no flip directo

            elif position == -1:
                if short_signal:
                    target_position = -1
                elif flat_signal:
                    target_position = 0
                elif long_signal:
                    target_position = 0  # no flip directo

        fee_cost = cost_for_transition(position, target_position, cfg["fee_rate"], cfg["slippage_bps"])

        if target_position != position:
            n_trades += 1
            total_cost += fee_cost
            cooldown = cfg["cooldown_bars"]

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
        "final_capital": float(equity_curve[-1]) if len(equity_curve) else cfg["initial_capital"],
        "total_return": float(equity_curve[-1] / cfg["initial_capital"] - 1.0) if len(equity_curve) else 0.0,
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
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


def score_result(row: dict) -> float:
    """
    Score práctico para ordenar configuraciones.
    Favorece retorno/Sharpe y penaliza DD extremo e infra/sobre-exposición.
    """
    score = 0.0
    score += 3.0 * row["total_return"]
    score += 0.5 * row["sharpe_ratio"]
    score -= 2.0 * abs(min(row["max_drawdown"], 0.0))

    # Penalizar muy poca exposición
    if row["pct_time_in_market"] < 0.01:
        score -= 0.50
    elif row["pct_time_in_market"] < 0.02:
        score -= 0.20

    # Penalizar demasiados trades
    if row["n_trades"] > 200:
        score -= 0.30
    elif row["n_trades"] > 100:
        score -= 0.10

    return score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("SWEEP DE CALIBRACIÓN")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")

    data_path = root_dir / "data" / "processed" / "dataset_btc_context_1h.csv"
    model_path = root_dir / "artifacts" / "models" / "gru_alpha_model.pt"
    scaler_path = root_dir / "artifacts" / "scalers" / "gru_alpha_scaler.json"

    if not data_path.exists():
        raise FileNotFoundError(f"No existe {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No existe {scaler_path}")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_regime"] = infer_regime_labels(df)

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler_payload = json.load(f)

    feature_cols = scaler_payload["feature_columns"]
    df_test = build_test_split(df)
    features_scaled = apply_scaler(df_test, feature_cols, scaler_payload)

    ckpt = torch.load(model_path, map_location=device)
    model_config = GRUAlphaConfig(**ckpt["model_config"])
    model = GRUAlphaModel(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    keys = list(SWEEP_GRID.keys())
    values = [SWEEP_GRID[k] for k in keys]
    combos = list(itertools.product(*values))

    logger.info(f"Total configuraciones a evaluar: {len(combos)}")

    results = []

    for i, combo in enumerate(combos, start=1):
        cfg = BASE_CONFIG.copy()
        for k, v in zip(keys, combo):
            cfg[k] = v

        metrics = run_backtest(
            df=df_test,
            features_scaled=features_scaled,
            model=model,
            seq_len=cfg["seq_len"],
            device=device,
            cfg=cfg,
        )

        row = {**cfg, **metrics}
        row["ranking_score"] = score_result(row)
        results.append(row)

        logger.info(
            f"[{i:03d}/{len(combos):03d}] "
            f"thr={cfg['threshold_long']:.4f} | "
            f"cd={cfg['cooldown_bars']} | hold={cfg['min_hold_bars']} | "
            f"conf={cfg['min_direction_confidence']:.2f} | "
            f"edge={cfg['min_edge_over_cost']:.4f} || "
            f"ret={metrics['total_return']:.4f} | "
            f"dd={metrics['max_drawdown']:.4f} | "
            f"sh={metrics['sharpe_ratio']:.3f} | "
            f"trades={metrics['n_trades']} | "
            f"expo={metrics['pct_time_in_market']:.3f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["ranking_score", "total_return", "sharpe_ratio"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    reports_dir = root_dir / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_path = reports_dir / "sweep_calibration_results.csv"
    top_path = reports_dir / "sweep_calibration_top20.csv"

    results_df.to_csv(out_path, index=False)
    results_df.head(20).to_csv(top_path, index=False)

    logger.info("=" * 70)
    logger.info("TOP 10 CONFIGURACIONES")
    logger.info("=" * 70)
    cols = [
        "ranking_score",
        "threshold_long",
        "cooldown_bars",
        "min_hold_bars",
        "min_direction_confidence",
        "min_edge_over_cost",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "n_trades",
        "pct_time_in_market",
    ]
    for _, row in results_df.head(10)[cols].iterrows():
        logger.info(row.to_dict())

    logger.info(f"Resultados completos: {out_path}")
    logger.info(f"Top 20: {top_path}")


if __name__ == "__main__":
    main()
