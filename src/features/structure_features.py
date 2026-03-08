from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a, b, eps: float = 1e-8):
    return a / (b + eps)


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features de estructura de mercado inspiradas en cómo opera un trader:
    - impulso
    - compresión / expansión
    - posición dentro del rango
    - fuerza de vela
    - eficiencia de tendencia
    - pullback desde swing high / swing low
    - proxies simples de doble suelo / doble techo

    Todas las features usan SOLO información pasada o presente.
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # ATR simple si no existe
    if "atr" in df.columns:
        atr = df["atr"].copy()
    else:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

    # =========================
    # IMPULSO
    # =========================
    df["impulse_3"] = close.pct_change(3)
    df["impulse_6"] = close.pct_change(6)
    df["impulse_12"] = close.pct_change(12)

    # impulso normalizado por ATR
    df["impulse_atr_6"] = _safe_div(close - close.shift(6), atr)
    df["impulse_atr_12"] = _safe_div(close - close.shift(12), atr)

    # =========================
    # COMPRESIÓN / EXPANSIÓN
    # =========================
    rv_12 = close.pct_change().rolling(12).std()
    rv_48 = close.pct_change().rolling(48).std()
    df["realized_vol_12"] = rv_12
    df["realized_vol_48"] = rv_48
    df["compression_ratio"] = _safe_div(rv_12, rv_48)

    range_12 = (high.rolling(12).max() - low.rolling(12).min())
    range_48 = (high.rolling(48).max() - low.rolling(48).min())
    df["range_compression"] = _safe_div(range_12, range_48)

    # =========================
    # POSICIÓN EN EL RANGO
    # =========================
    high_12 = high.rolling(12).max()
    low_12 = low.rolling(12).min()
    high_24 = high.rolling(24).max()
    low_24 = low.rolling(24).min()

    df["range_pos_12"] = _safe_div(close - low_12, high_12 - low_12)
    df["range_pos_24"] = _safe_div(close - low_24, high_24 - low_24)

    df["dist_to_high_12_atr"] = _safe_div(high_12 - close, atr)
    df["dist_to_low_12_atr"] = _safe_div(close - low_12, atr)
    df["dist_to_high_24_atr"] = _safe_div(high_24 - close, atr)
    df["dist_to_low_24_atr"] = _safe_div(close - low_24, atr)

    # =========================
    # FUERZA DE VELA
    # =========================
    candle_range = (high - low).replace(0, np.nan)
    body = close - open_

    df["body_strength"] = _safe_div(body, candle_range)
    df["upper_wick_ratio"] = _safe_div(high - np.maximum(open_, close), candle_range)
    df["lower_wick_ratio"] = _safe_div(np.minimum(open_, close) - low, candle_range)

    # =========================
    # EFICIENCIA DE TENDENCIA
    # =========================
    directional_move_12 = (close - close.shift(12)).abs()
    path_length_12 = close.diff().abs().rolling(12).sum()
    df["trend_efficiency_12"] = _safe_div(directional_move_12, path_length_12)

    directional_move_24 = (close - close.shift(24)).abs()
    path_length_24 = close.diff().abs().rolling(24).sum()
    df["trend_efficiency_24"] = _safe_div(directional_move_24, path_length_24)

    # =========================
    # PULLBACK / EXTENSIÓN
    # =========================
    recent_high_20 = high.rolling(20).max()
    recent_low_20 = low.rolling(20).min()

    df["pullback_from_high_atr"] = _safe_div(recent_high_20 - close, atr)
    df["bounce_from_low_atr"] = _safe_div(close - recent_low_20, atr)

    # =========================
    # VOLUMEN CONTEXTUAL
    # =========================
    vol_ma_20 = volume.rolling(20).mean()
    df["volume_zscore_20"] = _safe_div(volume - vol_ma_20, volume.rolling(20).std())
    df["volume_ratio_20"] = _safe_div(volume, vol_ma_20)

    # =========================
    # PROXIES SIMPLES DE DOBLE TECHO / DOBLE SUELO
    # No son figuras “de libro”, sino estados estructurales cuantificables
    # =========================
    low_lookback = low.rolling(30).min()
    high_lookback = high.rolling(30).max()

    # cercanía a suelo / techo previo
    df["near_prev_low"] = ((close - low_lookback).abs() <= 0.5 * atr).astype(float)
    df["near_prev_high"] = ((high_lookback - close).abs() <= 0.5 * atr).astype(float)

    # rechazo desde suelo / techo
    df["bull_rejection"] = ((df["lower_wick_ratio"] > 0.4) & (body > 0)).astype(float)
    df["bear_rejection"] = ((df["upper_wick_ratio"] > 0.4) & (body < 0)).astype(float)

    # proxy doble suelo / doble techo
    df["double_bottom_proxy"] = (df["near_prev_low"] * df["bull_rejection"]).astype(float)
    df["double_top_proxy"] = (df["near_prev_high"] * df["bear_rejection"]).astype(float)

    # limpieza
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    return df
