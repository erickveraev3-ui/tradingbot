from __future__ import annotations

import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Detecta swing highs y swing lows simples usando una ventana local.
    """
    out = df.copy()

    highs = out["high"].values
    lows = out["low"].values

    swing_high = np.zeros(len(out), dtype=np.int8)
    swing_low = np.zeros(len(out), dtype=np.int8)

    for i in range(window, len(out) - window):
        if highs[i] == np.max(highs[i - window:i + window + 1]):
            swing_high[i] = 1

        if lows[i] == np.min(lows[i - window:i + window + 1]):
            swing_low[i] = 1

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out


def build_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features de estructura:
    - último swing high/low
    - distancia a swing high/low
    - ruptura de swing high/low
    """
    out = df.copy()

    out["last_swing_high"] = out["high"].where(out["swing_high"] == 1).ffill()
    out["last_swing_low"] = out["low"].where(out["swing_low"] == 1).ffill()

    close = out["close"].replace(0, np.nan)

    out["dist_to_swing_high"] = (out["last_swing_high"] - out["close"]) / close
    out["dist_to_swing_low"] = (out["close"] - out["last_swing_low"]) / close

    out["break_swing_high"] = (out["close"] > out["last_swing_high"]).astype(int)
    out["break_swing_low"] = (out["close"] < out["last_swing_low"]).astype(int)

    return out


def detect_double_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta proxies simples de doble techo / doble suelo.
    """
    out = df.copy()

    # doble techo: swing high muy cerca del máximo de las últimas 20 velas
    out["double_top"] = (
        (out["swing_high"] == 1) &
        ((out["high"].rolling(20).max() - out["high"]).abs() / out["close"] < 0.002)
    ).astype(int)

    # doble suelo: swing low muy cerca del mínimo de las últimas 20 velas
    out["double_bottom"] = (
        (out["swing_low"] == 1) &
        ((out["low"] - out["low"].rolling(20).min()).abs() / out["close"] < 0.002)
    ).astype(int)

    return out


def structure_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score estructural simple:
    - break_swing_high favorece continuidad alcista
    - break_swing_low favorece continuidad bajista
    - double_bottom favorece reversión alcista
    - double_top favorece reversión bajista
    """
    out = df.copy()

    out["structure_score_long"] = (
        1.5 * out["break_swing_high"] +
        2.0 * out["double_bottom"]
    )

    out["structure_score_short"] = (
        1.5 * out["break_swing_low"] +
        2.0 * out["double_top"]
    )

    return out


def add_structure_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Wrapper principal.
    """
    out = df.copy()
    out = detect_swings(out, window=window)
    out = build_market_structure(out)
    out = detect_double_patterns(out)
    out = structure_score(out)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().bfill()

    return out
