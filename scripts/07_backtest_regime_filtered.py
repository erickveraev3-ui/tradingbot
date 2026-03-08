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

from src.models.gru_alpha_model import GRUAlphaModel, GRUAlphaConfig


CONFIG = {
    "seq_len": 64,
    "initial_capital": 10000.0,

    # Costes
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,

    # Señal base
    "threshold_long": 0.0035,
    "threshold_short": -0.0045,
    "min_edge_over_cost": 0.0010,

    # Filtro probabilístico
    "min_direction_confidence": 0.56,
    "min_prob_spread": 0.12,   # |p_long - p_short|

    # Ejecución
    "cooldown_bars": 3,
    "min_hold_bars": 3,
    "use_short": False,

    # Filtro de régimen
    "allowed_regimes_long": [0, 2],
    "allowed_regimes_short": [1],

    # Filtro de volatilidad
    "min_vol_ratio": 0.60,
    "max_vol_ratio": 1.80,
    "max_abs_bb_position": 1.80,

    # Position sizing
    "base_position": 0.25,     # 25% mínimo cuando hay señal válida
    "max_position": 1.00,      # 100% máximo
    "edge_scale": 120.0,       # convierte edge pequeño en tamaño gradual
    "regime_mult_long": {
        0: 1.00,   # trend up
        1: 0.00,   # trend down (no long)
        2: 0.70,   # mean reversion
        3: 0.00,   # neutral/noise -> no trade
    },
    "regime_mult_short": {
        0: 0.00,
        1: 1.00,
        2: 0.60,
        3: 0.00,
    },
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
    edge: float,
    confidence: float,
    prob_spread: float,
    regime: int,
) -> float:
    """
    Position sizing dinámico:
    - base mínima si la señal es válida
    - aumenta con edge y confianza
    - se modula por régimen
    """
    if direction == 0:
        return 0.0

    if direction > 0:
        regime_mult = CONFIG["regime_mult_long"].get(regime, 0.0)
    else:
        regime_mult = CONFIG["regime_mult_short"].get(regime, 0.0)

    if regime_mult <= 0:
        return 0.0

    # confianza relativa a partir del mínimo exigido
    conf_excess = max(0.0, confidence - CONFIG["min_direction_confidence"])
    spread_excess = max(0.0, prob_spread - CONFIG["min_prob_spread"])
    edge_scaled = max(0.0, edge * CONFIG["edge_scale"])

    size = CONFIG["base_position"]
    size += 1.50 * conf_excess
    size += 1.20 * spread_excess
    size += edge_scaled

    size *= regime_mult
    size = min(size, CONFIG["max_position"])
    size = max(0.0, size)

    return size * float(direction)


def run_backtest(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    model: GRUAlphaModel,
    seq_len: int,
    device: str,
):
    capital = CONFIG["initial_capital"]
    equity_curve = []
    trade_log = []
    diagnostics = []

    position = 0.0
    cooldown = 0
    hold_bars = 0

    model.eval()

    strategy_returns = []

    for idx in range(seq_len - 1, len(df) - 1):
        row = df.iloc[idx]
        close_now = float(row["close"])
        close_next = float(df.iloc[idx + 1]["close"])
        ts = row["timestamp"]

        vol_ratio = float(row["vol_ratio"]) if "vol_ratio" in df.columns and pd.notna(row["vol_ratio"]) else 1.0
        bb_position = float(row["bb_position"]) if "bb_position" in df.columns and pd.notna(row["bb_position"]) else 0.0

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

        prob_spread = abs(p_long - p_short)
        score = 0.70 * pred_ret_1h + 0.30 * pred_ret_4h

        # Filtro estructural de volatilidad/contexto
        vol_filter_ok = (
            CONFIG["min_vol_ratio"] <= vol_ratio <= CONFIG["max_vol_ratio"]
            and abs(bb_position) <= CONFIG["max_abs_bb_position"]
        )

        # Coste estimado de entrada desde flat (más estable para filtrar edge)
        est_entry_cost = cost_for_transition(0.0, 1.0, CONFIG["fee_rate"], CONFIG["slippage_bps"])
        est_edge_long = score - est_entry_cost
        est_edge_short = -score - est_entry_cost

        long_signal = (
            vol_filter_ok
            and regime in CONFIG["allowed_regimes_long"]
            and score >= CONFIG["threshold_long"]
            and p_long >= CONFIG["min_direction_confidence"]
            and prob_spread >= CONFIG["min_prob_spread"]
            and est_edge_long >= CONFIG["min_edge_over_cost"]
        )

        short_signal = (
            CONFIG["use_short"]
            and vol_filter_ok
            and regime in CONFIG["allowed_regimes_short"]
            and score <= CONFIG["threshold_short"]
            and p_short >= CONFIG["min_direction_confidence"]
            and prob_spread >= CONFIG["min_prob_spread"]
            and est_edge_short >= CONFIG["min_edge_over_cost"]
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
                        edge=est_edge_long,
                        confidence=p_long,
                        prob_spread=prob_spread,
                        regime=regime,
                    )
                    reason = "open_long"
                elif short_signal:
                    target_position = compute_position_size(
                        direction=-1,
                        edge=est_edge_short,
                        confidence=p_short,
                        prob_spread=prob_spread,
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
                        edge=est_edge_long,
                        confidence=p_long,
                        prob_spread=prob_spread,
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
                        edge=est_edge_short,
                        confidence=p_short,
                        prob_spread=prob_spread,
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
                "timestamp": ts,
                "close": close_now,
                "score": score,
                "pred_ret_1h": pred_ret_1h,
                "pred_ret_4h": pred_ret_4h,
                "p_long": p_long,
                "p_short": p_short,
                "prob_spread": prob_spread,
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
            "timestamp": ts,
            "score": score,
            "pred_ret_1h": pred_ret_1h,
            "pred_ret_4h": pred_ret_4h,
            "p_long": p_long,
            "p_short": p_short,
            "prob_spread": prob_spread,
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
        "timestamp": df.iloc[seq_len - 1: len(df) - 1]["timestamp"].values,
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
        "avg_prob_spread": float(diag_df["prob_spread"].mean()) if len(diag_df) else 0.0,
        "avg_vol_ratio": float(diag_df["vol_ratio"].mean()) if len(diag_df) else 0.0,
    }

    return bt, metrics, trades_df, diag_df


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("BACKTEST REGIME FILTERED")
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

    bt, metrics, trades_df, diag_df = run_backtest(
        df=df_test,
        features_scaled=features_scaled,
        model=model,
        seq_len=CONFIG["seq_len"],
        device=device,
    )

    reports_dir = root_dir / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    equity_path = reports_dir / "backtest_equity_regime_filtered.csv"
    trades_path = reports_dir / "backtest_trades_regime_filtered.csv"
    diag_path = reports_dir / "backtest_diagnostics_regime_filtered.csv"
    metrics_path = reports_dir / "backtest_metrics_regime_filtered.json"

    bt.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    diag_df.to_csv(diag_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("RESULTADOS BACKTEST REGIME FILTERED")
    logger.info("=" * 70)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    logger.info(f"Equity curve guardada en: {equity_path}")
    logger.info(f"Trades guardados en: {trades_path}")
    logger.info(f"Diagnóstico guardado en: {diag_path}")
    logger.info(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
