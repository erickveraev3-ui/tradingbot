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
    "fee_rate": 0.0006,
    "slippage_bps": 5.0,              # 0.05%
    "threshold_long": 0.0035,         # más exigente
    "threshold_short": -0.0040,       # aún más exigente para short
    "cooldown_bars": 6,
    "min_hold_bars": 4,
    "max_position": 1.0,
    "use_short": False,               # empezar en long-only
    "min_direction_confidence": 0.58, # p(long) o p(short)
    "min_edge_over_cost": 0.0012,     # retorno esperado neto mínimo
    "allowed_regimes_long": [0, 2],   # trend_up, mean_reverting
    "allowed_regimes_short": [1],     # si activas shorts
}


NON_FEATURE_COLUMNS = set([
    "timestamp",
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades",
    "target_return_1h",
    "target_return_4h",
    "target_direction",
])


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

    position = 0                 # -1 / 0 / +1
    cooldown = 0
    hold_bars = 0

    model.eval()

    timestamps = []
    strategy_returns = []
    diagnostics = []

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
        ts = df.iloc[idx]["timestamp"]

        # score más conservador
        score = 0.70 * pred_ret_1h + 0.30 * pred_ret_4h

        target_position = position
        action_reason = "hold"

        est_entry_cost = cost_for_transition(position, 1 if score > 0 else -1, CONFIG["fee_rate"], CONFIG["slippage_bps"])
        est_edge_long = score - est_entry_cost
        est_edge_short = -score - est_entry_cost

        long_signal = (
            score >= CONFIG["threshold_long"]
            and p_long >= CONFIG["min_direction_confidence"]
            and est_edge_long >= CONFIG["min_edge_over_cost"]
            and regime in CONFIG["allowed_regimes_long"]
        )

        short_signal = (
            CONFIG["use_short"]
            and score <= CONFIG["threshold_short"]
            and p_short >= CONFIG["min_direction_confidence"]
            and est_edge_short >= CONFIG["min_edge_over_cost"]
            and regime in CONFIG["allowed_regimes_short"]
        )

        flat_signal = not long_signal and not short_signal

        # gestión de cooldown
        if cooldown > 0:
            cooldown -= 1

        # gestión min_hold
        if position != 0:
            hold_bars += 1
        else:
            hold_bars = 0

        # reglas de decisión
        if cooldown > 0:
            target_position = position
            action_reason = "cooldown"

        elif position != 0 and hold_bars < CONFIG["min_hold_bars"]:
            target_position = position
            action_reason = "min_hold"

        else:
            if position == 0:
                if long_signal:
                    target_position = 1
                    action_reason = "open_long"
                elif short_signal:
                    target_position = -1
                    action_reason = "open_short"
                else:
                    target_position = 0
                    action_reason = "stay_flat"

            elif position == 1:
                if long_signal:
                    target_position = 1
                    action_reason = "keep_long"
                elif flat_signal:
                    target_position = 0
                    action_reason = "close_long"
                elif short_signal:
                    # no flip directo: cerramos primero
                    target_position = 0
                    action_reason = "close_long_no_flip"

            elif position == -1:
                if short_signal:
                    target_position = -1
                    action_reason = "keep_short"
                elif flat_signal:
                    target_position = 0
                    action_reason = "close_short"
                elif long_signal:
                    target_position = 0
                    action_reason = "close_short_no_flip"

        fee_cost = cost_for_transition(position, target_position, CONFIG["fee_rate"], CONFIG["slippage_bps"])

        if target_position != position:
            cooldown = CONFIG["cooldown_bars"]
            trade_log.append({
                "timestamp": ts,
                "close": close_now,
                "score": score,
                "pred_ret_1h": pred_ret_1h,
                "pred_ret_4h": pred_ret_4h,
                "p_long": p_long,
                "p_short": p_short,
                "regime": regime,
                "pos_prev": position,
                "pos_new": target_position,
                "cost": fee_cost,
                "reason": action_reason,
            })
            if target_position == 0:
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
            "regime": regime,
            "position": target_position,
            "reason": action_reason,
            "bar_return": raw_ret,
            "strategy_return": strat_ret,
            "equity": capital,
        })

        position = target_position
        timestamps.append(ts)
        strategy_returns.append(strat_ret)
        equity_curve.append(capital)

    bt = pd.DataFrame({
        "timestamp": timestamps,
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
        "long_trades": int((trades_df["pos_new"] == 1).sum()) if len(trades_df) else 0,
        "short_trades": int((trades_df["pos_new"] == -1).sum()) if len(trades_df) else 0,
        "flat_transitions": int((trades_df["pos_new"] == 0).sum()) if len(trades_df) else 0,
        "win_rate_bar": float((bt["strategy_return"] > 0).mean()) if len(bt) else 0.0,
        "mean_bar_return": float(bt["strategy_return"].mean()) if len(bt) else 0.0,
        "std_bar_return": float(bt["strategy_return"].std()) if len(bt) else 0.0,
        "avg_abs_score": float(diag_df["score"].abs().mean()) if len(diag_df) else 0.0,
        "pct_time_in_market": float((diag_df["position"] != 0).mean()) if len(diag_df) else 0.0,
        "pct_long": float((diag_df["position"] == 1).mean()) if len(diag_df) else 0.0,
        "pct_short": float((diag_df["position"] == -1).mean()) if len(diag_df) else 0.0,
    }

    return bt, metrics, trades_df, diag_df


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("BACKTEST SUPERVISADO")
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

    equity_path = reports_dir / "backtest_equity_supervised.csv"
    trades_path = reports_dir / "backtest_trades_supervised.csv"
    diag_path = reports_dir / "backtest_diagnostics_supervised.csv"
    metrics_path = reports_dir / "backtest_metrics_supervised.json"

    bt.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    diag_df.to_csv(diag_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("RESULTADOS BACKTEST")
    logger.info("=" * 70)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    logger.info(f"Equity curve guardada en: {equity_path}")
    logger.info(f"Trades guardados en: {trades_path}")
    logger.info(f"Diagnóstico guardado en: {diag_path}")
    logger.info(f"Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
