from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def compute_setup_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score de setup tipo trader profesional:
    combina probabilidad del meta-modelo + confluencia estructural.

    No queremos 40 métricas. Solo las que de verdad aportan:
    - probabilidad meta-model
    - fuerza de patrón
    - eficiencia de tendencia
    - ADX
    - compresión/squeeze
    """
    out = df.copy()

    prob_long = _safe_series(out, "prob_long", 0.0)
    prob_short = _safe_series(out, "prob_short", 0.0)

    pattern_long_score = _safe_series(out, "pattern_long_score", 0.0)
    pattern_short_score = _safe_series(out, "pattern_short_score", 0.0)

    trend_eff = _safe_series(out, "trend_efficiency_12", 0.0).clip(lower=0.0, upper=1.0)
    adx = _safe_series(out, "adx", 0.0).clip(lower=0.0, upper=1.0)
    compression = _safe_series(out, "compression_ratio", 1.0).clip(lower=0.0, upper=3.0)

    squeeze_up = _safe_series(out, "pattern_squeeze_break_up", 0.0)
    squeeze_down = _safe_series(out, "pattern_squeeze_break_down", 0.0)

    # bonus moderados, no excesivos
    long_confluence = (
        1.0
        + 0.35 * pattern_long_score.clip(upper=3.0)
        + 0.30 * trend_eff
        + 0.20 * adx
        + 0.20 * squeeze_up
        + 0.10 * (compression < 1.0).astype(float)
    )

    short_confluence = (
        1.0
        + 0.35 * pattern_short_score.clip(upper=3.0)
        + 0.30 * trend_eff
        + 0.20 * adx
        + 0.20 * squeeze_down
        + 0.10 * (compression < 1.0).astype(float)
    )

    # score principal
    out["setup_score_long"] = prob_long * long_confluence
    out["setup_score_short"] = prob_short * short_confluence

    # ventaja relativa entre lados
    out["setup_edge_long"] = out["setup_score_long"] - out["setup_score_short"]
    out["setup_edge_short"] = out["setup_score_short"] - out["setup_score_long"]

    return out


def select_top_setups(
    df: pd.DataFrame,
    side: str,
    top_percent: float,
    candidate_col: str,
) -> pd.DataFrame:
    """
    Selecciona el top X% de setups dentro de los candidatos de ese lado.
    """
    if side not in {"long", "short"}:
        raise ValueError("side debe ser 'long' o 'short'")

    out = df.copy()
    score_col = "setup_score_long" if side == "long" else "setup_score_short"
    flag_col = f"selected_{side}"

    candidate_mask = out[candidate_col] == 1
    candidate_scores = out.loc[candidate_mask, score_col]

    if len(candidate_scores) == 0:
        out[flag_col] = 0
        return out

    q = max(0.0, min(1.0, 1.0 - top_percent))
    threshold = candidate_scores.quantile(q)

    out[flag_col] = (
        candidate_mask
        & (out[score_col] >= threshold)
    ).astype(int)

    return out
