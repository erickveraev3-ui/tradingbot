from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
from loguru import logger

from src.features.feature_builder import FeatureBuilder, load_csv
from src.features.structure_features import add_structure_features
from src.features.pattern_engine import add_pattern_engine_features
from src.structure.swing_structure import add_structure_features as add_swing_structure_features


RAW_DIR = root_dir / "data" / "raw"
PROCESSED_DIR = root_dir / "data" / "processed"


CONFIG = {
    # triple barrier
    "tp_atr_mult_long": 1.8,
    "sl_atr_mult_long": 1.2,
    "tp_atr_mult_short": 1.8,
    "sl_atr_mult_short": 1.2,
    "max_holding_bars": 12,

    # thresholds candidatos
    "min_pattern_long_score": 1.0,
    "min_pattern_short_score": 1.0,
    "min_adx_trend": 0.18,
}


def validate_dataframe(df: pd.DataFrame, name: str):
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{name}: falta columna {col}")

    if df.empty:
        raise ValueError(f"{name}: dataframe vacío")

    if df["timestamp"].duplicated().any():
        raise ValueError(f"{name}: timestamps duplicados")

    df.sort_values("timestamp", inplace=True)


def align_dataframes(btc: pd.DataFrame, eth: pd.DataFrame, sol: pd.DataFrame):
    common_ts = (
        set(btc["timestamp"])
        .intersection(set(eth["timestamp"]))
        .intersection(set(sol["timestamp"]))
    )
    common_ts = sorted(common_ts)

    btc = btc[btc["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    eth = eth[eth["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    sol = sol[sol["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)

    return btc, eth, sol


def build_candidate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    adx = df["adx"] if "adx" in df.columns else pd.Series(0.0, index=df.index)
    ema_trend = df["ema_trend"] if "ema_trend" in df.columns else pd.Series(0.0, index=df.index)

    pattern_long_score = df["pattern_long_score"] if "pattern_long_score" in df.columns else pd.Series(0.0, index=df.index)
    pattern_short_score = df["pattern_short_score"] if "pattern_short_score" in df.columns else pd.Series(0.0, index=df.index)

    squeeze_up = df["pattern_squeeze_break_up"] if "pattern_squeeze_break_up" in df.columns else pd.Series(0.0, index=df.index)
    squeeze_down = df["pattern_squeeze_break_down"] if "pattern_squeeze_break_down" in df.columns else pd.Series(0.0, index=df.index)

    db_confirm = df["double_bottom_break_confirm"] if "double_bottom_break_confirm" in df.columns else pd.Series(0.0, index=df.index)
    dt_confirm = df["double_top_break_confirm"] if "double_top_break_confirm" in df.columns else pd.Series(0.0, index=df.index)

    sweep_long = df["pattern_liquidity_sweep_long"] if "pattern_liquidity_sweep_long" in df.columns else pd.Series(0.0, index=df.index)
    sweep_short = df["pattern_liquidity_sweep_short"] if "pattern_liquidity_sweep_short" in df.columns else pd.Series(0.0, index=df.index)

    trend_long = (ema_trend > 0) & (adx >= CONFIG["min_adx_trend"])
    trend_short = (ema_trend < 0) & (adx >= CONFIG["min_adx_trend"])

    candidate_long = (
        (pattern_long_score >= CONFIG["min_pattern_long_score"])
        & (
            trend_long
            | (db_confirm > 0)
            | (squeeze_up > 0)
            | (sweep_long > 0)
        )
    ).astype(int)

    candidate_short = (
        (pattern_short_score >= CONFIG["min_pattern_short_score"])
        & (
            trend_short
            | (dt_confirm > 0)
            | (squeeze_down > 0)
            | (sweep_short > 0)
        )
    ).astype(int)

    df["candidate_long"] = candidate_long
    df["candidate_short"] = candidate_short
    return df


def triple_barrier_outcome(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
    direction: str,
    tp_mult: float,
    sl_mult: float,
    max_holding_bars: int,
):
    n = len(close_arr)
    labels = np.zeros(n, dtype=np.int8)
    realized_returns = np.zeros(n, dtype=np.float32)
    hit_bars = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        entry = close_arr[i]
        atr = atr_arr[i]

        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
            continue

        if direction == "long":
            tp = entry + tp_mult * atr
            sl = entry - sl_mult * atr
        else:
            tp = entry - tp_mult * atr
            sl = entry + sl_mult * atr

        end = min(n - 1, i + max_holding_bars)
        if i + 1 > end:
            continue

        decided = False

        for j in range(i + 1, end + 1):
            hi = high_arr[j]
            lo = low_arr[j]

            if direction == "long":
                hit_tp = hi >= tp
                hit_sl = lo <= sl
            else:
                hit_tp = lo <= tp
                hit_sl = hi >= sl

            if hit_tp and hit_sl:
                labels[i] = -1
                hit_bars[i] = j
                realized_returns[i] = (sl / entry - 1.0) if direction == "long" else (entry / sl - 1.0)
                decided = True
                break

            if hit_tp:
                labels[i] = 1
                hit_bars[i] = j
                realized_returns[i] = (tp / entry - 1.0) if direction == "long" else (entry / tp - 1.0)
                decided = True
                break

            if hit_sl:
                labels[i] = -1
                hit_bars[i] = j
                realized_returns[i] = (sl / entry - 1.0) if direction == "long" else (entry / sl - 1.0)
                decided = True
                break

        if not decided:
            expiry_close = close_arr[end]
            labels[i] = 0
            hit_bars[i] = end
            realized_returns[i] = (expiry_close / entry - 1.0) if direction == "long" else (entry / expiry_close - 1.0)

    return labels, realized_returns, hit_bars


def apply_candidate_mask(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mask_long = df["candidate_long"].values.astype(bool)
    mask_short = df["candidate_short"].values.astype(bool)

    df.loc[~mask_long, "tb_long_label"] = 0
    df.loc[~mask_long, "tb_long_win"] = 0
    df.loc[~mask_long, "tb_long_loss"] = 0

    df.loc[~mask_short, "tb_short_label"] = 0
    df.loc[~mask_short, "tb_short_win"] = 0
    df.loc[~mask_short, "tb_short_loss"] = 0

    return df


def main():
    logger.info("=" * 80)
    logger.info("CREANDO DATASET TRIPLE-BARRIER 15M CON CONTEXTO EXÓGENO REAL")
    logger.info("=" * 80)

    btc_path = RAW_DIR / "btcusdt_15m.csv"
    eth_path = RAW_DIR / "ethusdt_15m.csv"
    sol_path = RAW_DIR / "solusdt_15m.csv"

    for p in [btc_path, eth_path, sol_path]:
        if not p.exists():
            raise FileNotFoundError(f"No existe {p}")

    btc = load_csv(str(btc_path))
    eth = load_csv(str(eth_path))
    sol = load_csv(str(sol_path))

    validate_dataframe(btc, "BTC")
    validate_dataframe(eth, "ETH")
    validate_dataframe(sol, "SOL")

    btc, eth, sol = align_dataframes(btc, eth, sol)
    logger.info(f"Filas alineadas: {len(btc):,}")

    builder = FeatureBuilder()
    df = builder.build(btc, eth, sol)

    df = add_structure_features(df)
    df = add_pattern_engine_features(df)
    df = add_swing_structure_features(df)
    df = build_candidate_signals(df)

    if "atr" not in df.columns:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean().ffill().bfill()

    high_arr = df["high"].values.astype(np.float64)
    low_arr = df["low"].values.astype(np.float64)
    close_arr = df["close"].values.astype(np.float64)
    atr_arr = df["atr"].values.astype(np.float64)

    long_labels, long_returns, long_hit_bars = triple_barrier_outcome(
        high_arr, low_arr, close_arr, atr_arr,
        direction="long",
        tp_mult=CONFIG["tp_atr_mult_long"],
        sl_mult=CONFIG["sl_atr_mult_long"],
        max_holding_bars=CONFIG["max_holding_bars"],
    )

    short_labels, short_returns, short_hit_bars = triple_barrier_outcome(
        high_arr, low_arr, close_arr, atr_arr,
        direction="short",
        tp_mult=CONFIG["tp_atr_mult_short"],
        sl_mult=CONFIG["sl_atr_mult_short"],
        max_holding_bars=CONFIG["max_holding_bars"],
    )

    df["tb_long_label"] = long_labels
    df["tb_long_return"] = long_returns
    df["tb_long_hit_bar"] = long_hit_bars
    df["tb_long_win"] = (df["tb_long_label"] == 1).astype(int)
    df["tb_long_loss"] = (df["tb_long_label"] == -1).astype(int)

    df["tb_short_label"] = short_labels
    df["tb_short_return"] = short_returns
    df["tb_short_hit_bar"] = short_hit_bars
    df["tb_short_win"] = (df["tb_short_label"] == 1).astype(int)
    df["tb_short_loss"] = (df["tb_short_label"] == -1).astype(int)

    df = apply_candidate_mask(df)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset_btc_triple_barrier_15m.csv"
    df.to_csv(out_path, index=False)

    logger.info(f"Dataset guardado en: {out_path}")
    logger.info(f"Filas finales: {len(df):,}")
    logger.info(f"Columnas: {len(df.columns)}")

    logger.info("Resumen candidatos:")
    logger.info(f"candidate_long: {df['candidate_long'].mean():.2%}")
    logger.info(f"candidate_short: {df['candidate_short'].mean():.2%}")

    logger.info("Resumen triple-barrier:")
    logger.info(f"tb_long_win: {df['tb_long_win'].mean():.2%}")
    logger.info(f"tb_long_loss: {df['tb_long_loss'].mean():.2%}")
    logger.info(f"tb_short_win: {df['tb_short_win'].mean():.2%}")
    logger.info(f"tb_short_loss: {df['tb_short_loss'].mean():.2%}")

    logger.info("Scores de patrones:")
    logger.info(f"pattern_long_score mean: {df['pattern_long_score'].mean():.4f}")
    logger.info(f"pattern_short_score mean: {df['pattern_short_score'].mean():.4f}")


if __name__ == "__main__":
    main()