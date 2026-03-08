from __future__ import annotations

import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.models.gru_event_model import GRUEventModel, GRUEventConfig


EVENT_TARGETS = [
    "target_event_long",
    "target_event_short",
    "event_breakout_up",
    "event_breakdown_down",
]

CONFIG = {
    "seq_len": 96,
    "initial_capital": 10000.0,

    # Costes
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,

    # Dirección
    "use_short": True,
    "cooldown_bars": 3,
    "min_hold_bars": 3,

    # Filtro estructural
    "allowed_regimes_long": [0, 2],
    "allowed_regimes_short": [1, 2],
    "min_vol_ratio": 0.60,
    "max_vol_ratio": 1.90,
    "max_abs_bb_position": 1.90,

    # Calidad de señal
    "min_long_short_gap": 0.05,        # P(long)-P(short)
    "min_short_long_gap": 0.05,        # P(short)-P(long)
    "min_breakout_bonus": 0.55,
    "min_breakdown_bonus": 0.55,

    # Sizing
    "base_position": 0.20,
    "max_position": 1.00,
    "signal_scale": 1.80,
    "breakout_scale": 0.40,
    "breakdown_scale": 0.40,
    "regime_mult_long": {0: 1.00, 1: 0.00, 2: 0.65, 3: 0.00},
    "regime_mult_short": {0: 0.00, 1: 1.00, 2: 0.55, 3: 0.00},

    # Penalización de edge por coste esperado
    "min_edge_over_cost": 0.0,
}


def apply_robust_scaler(df: pd.DataFrame, feature_cols: list[str], scaler_payload: dict) -> np.ndarray:
    x = df[feature_cols].values.astype(np.float32)
    median = np.array(scaler_payload["median"], dtype=np.float32)
    scale = np.array(scaler_payload["scale"], dtype=np.float32) + 1e-8
    x = (x - median) / scale
    x = np.clip(x, -8, 8)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def apply_calibration_np(logits: np.ndarray, calibration_payload: dict) -> np.ndarray:
    log_temp = np.array(calibration_payload["log_temp"], dtype=np.float32)
    bias = np.array(calibration_payload["bias"], dtype=np.float32)
    temp = np.exp(log_temp).clip(1e-3, 100.0)
    logits_cal = (logits + bias) / temp
    probs = 1.0 / (1.0 + np.exp(-logits_cal))
    return probs


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


def build_test_split(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[val_end:].copy().reset_index(drop=True)


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


def cost_for_transition(pos_prev: float, pos_new: float, fee_rate: float, slippage_bps: float) -> float:
    turnover = abs(pos_new - pos_prev)
    if turnover == 0:
        return 0.0
    cost = turnover * fee_rate
    cost += turnover * (slippage_bps / 10000.0)
    return cost


def compute_position_size(
    direction: int,
    p_dir: float,
    p_opp: float,
    p_break: float,
    regime: int,
) -> float:
    if direction == 0:
        return 0.0

    gap = max(0.0, p_dir - p_opp)

    size = CONFIG["base_position"]
    size += CONFIG["signal_scale"] * gap

    if direction > 0 and p_break >= CONFIG["min_breakout_bonus"]:
        size += CONFIG["breakout_scale"] * (p_break - CONFIG["min_breakout_bonus"])

    if direction < 0 and p_break >= CONFIG["min_breakdown_bonus"]:
        size += CONFIG["breakdown_scale"] * (p_break - CONFIG["min_breakdown_bonus"])

    if direction > 0:
        size *= CONFIG["regime_mult_long"].get(regime, 0.0)
    else:
        size *= CONFIG["regime_mult_short"].get(regime, 0.0)

    size = min(size, CONFIG["max_position"])
    size = max(0.0, size)
    return float(direction) * size


def run_backtest(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    model: GRUEventModel,
    seq_len: int,
    device: str,
    calibration_payload: dict,
    thresholds: dict,
):
    capital = CONFIG["initial_capital"]
    position = 0.0
    cooldown = 0
    hold_bars = 0

    strategy_returns = []
    equity_curve = []
    trade_log = []
    diagnostics = []

    long_thr = thresholds["target_event_long"]["threshold"]
    short_thr = thresholds["target_event_short"]["threshold"]
    breakout_thr = thresholds["event_breakout_up"]["threshold"]
    breakdown_thr = thresholds["event_breakdown_down"]["threshold"]

    model.eval()

    for idx in range(seq_len - 1, len(df) - 1):
        row = df.iloc[idx]
        ts = row["timestamp"]
        close_now = float(row["close"])
        close_next = float(df.iloc[idx + 1]["close"])

        vol_ratio = float(row["vol_ratio"]) if "vol_ratio" in df.columns and pd.notna(row["vol_ratio"]) else 1.0
        bb_position = float(row["bb_position"]) if "bb_position" in df.columns and pd.notna(row["bb_position"]) else 0.0
        regime = int(row["target_regime"])

        x_seq = features_scaled[idx - seq_len + 1: idx + 1]
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(x_tensor)

        logits = out["event_logits"].cpu().numpy()
        probs = apply_calibration_np(logits, calibration_payload)[0]

        p_long = float(probs[0])
        p_short = float(probs[1])
        p_breakout = float(probs[2])
        p_breakdown = float(probs[3])

        vol_filter_ok = (
            CONFIG["min_vol_ratio"] <= vol_ratio <= CONFIG["max_vol_ratio"]
            and abs(bb_position) <= CONFIG["max_abs_bb_position"]
        )

        long_signal = (
            vol_filter_ok
            and regime in CONFIG["allowed_regimes_long"]
            and p_long >= long_thr
            and (p_long - p_short) >= CONFIG["min_long_short_gap"]
        )

        short_signal = (
            CONFIG["use_short"]
            and vol_filter_ok
            and regime in CONFIG["allowed_regimes_short"]
            and p_short >= short_thr
            and (p_short - p_long) >= CONFIG["min_short_long_gap"]
        )

        flat_signal = not long_signal and not short_signal

        if cooldown > 0:
            cooldown -= 1

        if abs(position) > 1e-12:
            hold_bars += 1
        else:
            hold_bars = 0

        target_position = position
        reason = "hold"

        if cooldown > 0:
            target_position = position
            reason = "cooldown"

        elif abs(position) > 1e-12 and hold_bars < CONFIG["min_hold_bars"]:
            target_position = position
            reason = "min_hold"

        else:
            if abs(position) < 1e-12:
                if long_signal:
                    target_position = compute_position_size(
                        direction=1,
                        p_dir=p_long,
                        p_opp=p_short,
                        p_break=p_breakout,
                        regime=regime,
                    )
                    reason = "open_long"
                elif short_signal:
                    target_position = compute_position_size(
                        direction=-1,
                        p_dir=p_short,
                        p_opp=p_long,
                        p_break=p_breakdown,
                        regime=regime,
                    )
                    reason = "open_short"
                else:
                    target_position = 0.0
                    reason = "stay_flat"

            elif position > 0:
                if long_signal:
                    target_position = compute_position_size(
                        direction=1,
                        p_dir=p_long,
                        p_opp=p_short,
                        p_break=p_breakout,
                        regime=regime,
                    )
                    reason = "resize_long"
                elif flat_signal:
                    target_position = 0.0
                    reason = "close_long"
                elif short_signal:
                    target_position = 0.0
                    reason = "close_long_no_flip"

            elif position < 0:
                if short_signal:
                    target_position = compute_position_size(
                        direction=-1,
                        p_dir=p_short,
                        p_opp=p_long,
                        p_break=p_breakdown,
                        regime=regime,
                    )
                    reason = "resize_short"
                elif flat_signal:
                    target_position = 0.0
                    reason = "close_short"
                elif long_signal:
                    target_position = 0.0
                    reason = "close_short_no_flip"

        fee_cost = cost_for_transition(position, target_position, CONFIG["fee_rate"], CONFIG["slippage_bps"])

        if abs(target_position - position) > 1e-12:
            trade_log.append({
                "timestamp": str(ts),
                "close": close_now,
                "p_long": p_long,
                "p_short": p_short,
                "p_breakout": p_breakout,
                "p_breakdown": p_breakdown,
                "regime": regime,
                "vol_ratio": vol_ratio,
                "bb_position": bb_position,
                "pos_prev": position,
                "pos_new": target_position,
                "cost": fee_cost,
                "reason": reason,
            })
            cooldown = CONFIG["cooldown_bars"]
            if abs(target_position) < 1e-12:
                hold_bars = 0

        raw_ret = (close_next / close_now) - 1.0
        strat_ret = target_position * raw_ret - fee_cost
        capital *= (1.0 + strat_ret)

        diagnostics.append({
            "timestamp": str(ts),
            "p_long": p_long,
            "p_short": p_short,
            "p_breakout": p_breakout,
            "p_breakdown": p_breakdown,
            "regime": regime,
            "vol_ratio": vol_ratio,
            "bb_position": bb_position,
            "position": target_position,
            "reason": reason,
            "bar_return": raw_ret,
            "strategy_return": strat_ret,
            "equity": capital,
        })

        position = target_position
        strategy_returns.append(strat_ret)
        equity_curve.append(capital)

    bt = pd.DataFrame({
        "timestamp": df.iloc[seq_len - 1: len(df) - 1]["timestamp"].astype(str).values,
        "strategy_return": strategy_returns,
        "equity": equity_curve,
    })

    trades_df = pd.DataFrame(trade_log)
    diag_df = pd.DataFrame(diagnostics)

    metrics = {
        "initial_capital": CONFIG["initial_capital"],
        "final_capital": float(bt["equity"].iloc[-1]) if len(bt) else CONFIG["initial_capital"],
        "total_return": float(bt["equity"].iloc[-1] / CONFIG["initial_capital"] - 1.0) if len(bt) else 0.0,
        "max_drawdown": max_drawdown(bt["equity"].values) if len(bt) else 0.0,
        "sharpe_ratio": sharpe_ratio(bt["strategy_return"].values) if len(bt) else 0.0,
        "sortino_ratio": sortino_ratio(bt["strategy_return"].values) if len(bt) else 0.0,
        "n_trades": int(len(trades_df)),
        "avg_trade_cost": float(trades_df["cost"].mean()) if len(trades_df) else 0.0,
        "win_rate_bar": float((bt["strategy_return"] > 0).mean()) if len(bt) else 0.0,
        "mean_bar_return": float(bt["strategy_return"].mean()) if len(bt) else 0.0,
        "std_bar_return": float(bt["strategy_return"].std()) if len(bt) else 0.0,
        "pct_time_in_market": float((diag_df["position"].abs() > 1e-12).mean()) if len(diag_df) else 0.0,
        "pct_long": float((diag_df["position"] > 1e-12).mean()) if len(diag_df) else 0.0,
        "pct_short": float((diag_df["position"] < -1e-12).mean()) if len(diag_df) else 0.0,
        "avg_abs_position": float(diag_df["position"].abs().mean()) if len(diag_df) else 0.0,
        "avg_p_long": float(diag_df["p_long"].mean()) if len(diag_df) else 0.0,
        "avg_p_short": float(diag_df["p_short"].mean()) if len(diag_df) else 0.0,
        "avg_p_breakout": float(diag_df["p_breakout"].mean()) if len(diag_df) else 0.0,
        "avg_p_breakdown": float(diag_df["p_breakdown"].mean()) if len(diag_df) else 0.0,
    }

    return bt, metrics, trades_df, diag_df


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 80)
    logger.info("BACKTEST DEL MODELO DE EVENTOS")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")

    data_path = root_dir / "data" / "processed" / "dataset_btc_event_1h.csv"
    model_path = root_dir / "artifacts" / "models" / "gru_event_model.pt"
    scaler_path = root_dir / "artifacts" / "scalers" / "gru_event_scaler.json"
    calibration_path = root_dir / "artifacts" / "models" / "gru_event_calibration.json"
    thresholds_path = root_dir / "artifacts" / "models" / "gru_event_thresholds.json"

    for p in [data_path, model_path, scaler_path, calibration_path, thresholds_path]:
        if not p.exists():
            raise FileNotFoundError(f"No existe {p}")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["target_regime"] = infer_regime_labels(df)

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler_payload = json.load(f)

    with open(calibration_path, "r", encoding="utf-8") as f:
        calibration_payload = json.load(f)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    feature_cols = scaler_payload["feature_columns"]
    df_test = build_test_split(df)
    features_scaled = apply_robust_scaler(df_test, feature_cols, scaler_payload)

    ckpt = torch.load(model_path, map_location=device)
    model_config = GRUEventConfig(**ckpt["model_config"])
    model = GRUEventModel(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bt, metrics, trades_df, diag_df = run_backtest(
        df=df_test,
        features_scaled=features_scaled,
        model=model,
        seq_len=CONFIG["seq_len"],
        device=device,
        calibration_payload=calibration_payload,
        thresholds=thresholds,
    )

    reports_dir = root_dir / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    equity_path = reports_dir / "backtest_equity_event_model.csv"
    trades_path = reports_dir / "backtest_trades_event_model.csv"
    diag_path = reports_dir / "backtest_diagnostics_event_model.csv"
    metrics_path = reports_dir / "backtest_metrics_event_model.json"

    bt.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    diag_df.to_csv(diag_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("=" * 80)
    logger.info("RESULTADOS BACKTEST EVENT MODEL")
    logger.info("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.6f}")
        else:
            logger.info(f"{k}: {v}")

    logger.info("Thresholds usados:")
    for k, v in thresholds.items():
        logger.info(f"{k}: {v}")

    logger.info(f"Equity curve guardada en: {equity_path}")
    logger.info(f"Trades guardados en: {trades_path}")
    logger.info(f"Diagnóstico guardado en: {diag_path}")
    logger.info(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
