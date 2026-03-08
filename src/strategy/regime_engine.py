from __future__ import annotations

import numpy as np
import pandas as pd


def infer_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regime engine robusto y autosuficiente.

    Detecta:
    - trend_up
    - trend_down
    - range

    Prioridad:
    1) usar ema_trend si existe
    2) si no existe, reconstruir con EMA20/EMA50 desde close
    """
    out = df.copy()

    if "close" not in out.columns:
        raise ValueError("Dataset necesita columna 'close'")

    close = out["close"].astype(float)

    if "ema_trend" in out.columns:
        trend_strength = out["ema_trend"].astype(float).fillna(0.0)
    else:
        ema_fast = close.ewm(span=20, adjust=False).mean()
        ema_slow = close.ewm(span=50, adjust=False).mean()
        trend_strength = ((ema_fast - ema_slow) / close).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # thresholds simples y realistas para BTC 1H
    trend_up = trend_strength > 0.0015
    trend_down = trend_strength < -0.0015

    regime = np.full(len(out), "range", dtype=object)
    regime[trend_up.values] = "trend_up"
    regime[trend_down.values] = "trend_down"

    out["regime_label"] = regime
    return out


def apply_regime_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta EV y setup score según régimen.
    """
    out = df.copy()

    if "regime_label" not in out.columns:
        raise ValueError("Falta 'regime_label'. Ejecuta infer_market_regime() antes.")

    regime = out["regime_label"]

    long_mult = np.ones(len(out), dtype=np.float32)
    short_mult = np.ones(len(out), dtype=np.float32)

    # favorecer el lado de la tendencia, penalizar el opuesto
    long_mult[regime == "trend_up"] = 1.20
    short_mult[regime == "trend_up"] = 0.75

    long_mult[regime == "trend_down"] = 0.75
    short_mult[regime == "trend_down"] = 1.20

    long_mult[regime == "range"] = 1.00
    short_mult[regime == "range"] = 1.00

    if "ev_long_net" in out.columns:
        out["ev_long_regime"] = out["ev_long_net"] * long_mult
    if "ev_short_net" in out.columns:
        out["ev_short_regime"] = out["ev_short_net"] * short_mult

    if "setup_score_long" in out.columns:
        out["setup_score_long_regime"] = out["setup_score_long"] * long_mult
    if "setup_score_short" in out.columns:
        out["setup_score_short_regime"] = out["setup_score_short"] * short_mult

    out["regime_mult_long"] = long_mult
    out["regime_mult_short"] = short_mult

    return out
