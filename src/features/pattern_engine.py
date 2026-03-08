from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a, b, eps: float = 1e-8):
    return a / (b + eps)


def _rolling_swing_low(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """
    Marca mínimos locales simples usando ventana centrada.
    Solo para feature engineering offline.
    """
    w = left + right + 1
    roll_min = low.rolling(w, center=True).min()
    return (low == roll_min).astype(float)


def _rolling_swing_high(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    w = left + right + 1
    roll_max = high.rolling(w, center=True).max()
    return (high == roll_max).astype(float)


def add_pattern_engine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features cuantitativas de patrones de trader:
    - doble suelo / doble techo
    - continuación de tendencia
    - squeeze + breakout readiness
    - liquidity sweep
    - ruptura confirmada

    No usa información futura salvo en swings centrados, por lo que:
    - es válido para dataset offline/research,
    - para live luego habrá que reemplazar por una versión causal.
    """
    df = df.copy()

    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    if "atr" in df.columns:
        atr = df["atr"].replace(0, np.nan).ffill().bfill()
    else:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().replace(0, np.nan).ffill().bfill()

    if "compression_ratio" in df.columns:
        compression = df["compression_ratio"]
    else:
        compression = close.pct_change().rolling(12).std() / (
            close.pct_change().rolling(48).std() + 1e-8
        )

    if "trend_efficiency_12" in df.columns:
        efficiency = df["trend_efficiency_12"]
    else:
        directional_move = (close - close.shift(12)).abs()
        path_length = close.diff().abs().rolling(12).sum()
        efficiency = _safe_div(directional_move, path_length)

    if "body_strength" in df.columns:
        body_strength = df["body_strength"]
    else:
        candle_range = (high - low).replace(0, np.nan)
        body_strength = _safe_div(close - open_, candle_range)

    # =========================================================
    # Swings
    # =========================================================
    df["swing_low_flag"] = _rolling_swing_low(low, 3, 3)
    df["swing_high_flag"] = _rolling_swing_high(high, 3, 3)

    last_swing_low = low.where(df["swing_low_flag"] > 0).ffill()
    prev_swing_low = last_swing_low.shift(1).where(df["swing_low_flag"] > 0).ffill()

    last_swing_high = high.where(df["swing_high_flag"] > 0).ffill()
    prev_swing_high = last_swing_high.shift(1).where(df["swing_high_flag"] > 0).ffill()

    df["last_swing_low"] = last_swing_low
    df["prev_swing_low"] = prev_swing_low
    df["last_swing_high"] = last_swing_high
    df["prev_swing_high"] = prev_swing_high

    # =========================================================
    # Double bottom / double top cuantitativo
    # =========================================================
    low_similarity = (last_swing_low - prev_swing_low).abs() <= (0.5 * atr)
    high_similarity = (last_swing_high - prev_swing_high).abs() <= (0.5 * atr)

    bounce_from_low = (close > last_swing_low + 0.5 * atr)
    rejection_up = body_strength > 0.2

    rejection_down = body_strength < -0.2
    drop_from_high = (close < last_swing_high - 0.5 * atr)

    df["pattern_double_bottom"] = (low_similarity & bounce_from_low & rejection_up).astype(float)
    df["pattern_double_top"] = (high_similarity & drop_from_high & rejection_down).astype(float)

    # neckline proxy
    rolling_mid_high = high.rolling(10).max()
    rolling_mid_low = low.rolling(10).min()

    df["double_bottom_break_confirm"] = (
        (df["pattern_double_bottom"] > 0)
        & (close > rolling_mid_high.shift(1))
    ).astype(float)

    df["double_top_break_confirm"] = (
        (df["pattern_double_top"] > 0)
        & (close < rolling_mid_low.shift(1))
    ).astype(float)

    # =========================================================
    # Tendencia + pullback + continuación
    # =========================================================
    if "ema_trend" in df.columns:
        ema_trend = df["ema_trend"]
    else:
        ema12 = close.ewm(span=12).mean()
        ema48 = close.ewm(span=48).mean()
        ema_trend = _safe_div(ema12 - ema48, close)

    if "pullback_from_high_atr" in df.columns:
        pullback_high = df["pullback_from_high_atr"]
    else:
        pullback_high = _safe_div(high.rolling(20).max() - close, atr)

    if "bounce_from_low_atr" in df.columns:
        bounce_low = df["bounce_from_low_atr"]
    else:
        bounce_low = _safe_div(close - low.rolling(20).min(), atr)

    trend_up = (ema_trend > 0.001) & (efficiency > 0.25)
    trend_down = (ema_trend < -0.001) & (efficiency > 0.25)

    healthy_pullback_long = (pullback_high > 0.4) & (pullback_high < 2.2)
    healthy_pullback_short = (bounce_low > 0.4) & (bounce_low < 2.2)

    df["pattern_trend_cont_long"] = (
        trend_up & healthy_pullback_long & (body_strength > 0)
    ).astype(float)

    df["pattern_trend_cont_short"] = (
        trend_down & healthy_pullback_short & (body_strength < 0)
    ).astype(float)

    # =========================================================
    # Squeeze
    # =========================================================
    df["pattern_squeeze"] = (
        (compression < 0.85) &
        (high.rolling(12).max() - low.rolling(12).min() < 4.0 * atr)
    ).astype(float)

    df["pattern_squeeze_break_up"] = (
        (df["pattern_squeeze"].shift(1) > 0)
        & (close > high.rolling(12).max().shift(1))
        & (volume > volume.rolling(20).mean())
    ).astype(float)

    df["pattern_squeeze_break_down"] = (
        (df["pattern_squeeze"].shift(1) > 0)
        & (close < low.rolling(12).min().shift(1))
        & (volume > volume.rolling(20).mean())
    ).astype(float)

    # =========================================================
    # Liquidity sweep
    # =========================================================
    prev_low = low.rolling(20).min().shift(1)
    prev_high = high.rolling(20).max().shift(1)

    # sweep alcista: barre mínimos y cierra por encima
    df["pattern_liquidity_sweep_long"] = (
        (low < prev_low)
        & (close > prev_low)
        & (body_strength > 0)
    ).astype(float)

    # sweep bajista: barre máximos y cierra por debajo
    df["pattern_liquidity_sweep_short"] = (
        (high > prev_high)
        & (close < prev_high)
        & (body_strength < 0)
    ).astype(float)

    # =========================================================
    # Composite scores
    # =========================================================
    df["pattern_long_score"] = (
        1.0 * df["pattern_double_bottom"] +
        1.2 * df["double_bottom_break_confirm"] +
        1.0 * df["pattern_trend_cont_long"] +
        1.1 * df["pattern_squeeze_break_up"] +
        0.9 * df["pattern_liquidity_sweep_long"]
    )

    df["pattern_short_score"] = (
        1.0 * df["pattern_double_top"] +
        1.2 * df["double_top_break_confirm"] +
        1.0 * df["pattern_trend_cont_short"] +
        1.1 * df["pattern_squeeze_break_down"] +
        0.9 * df["pattern_liquidity_sweep_short"]
    )

    # =========================================================
    # Limpieza
    # =========================================================
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    return df
