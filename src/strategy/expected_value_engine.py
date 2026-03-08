from __future__ import annotations

import numpy as np
import pandas as pd


def compute_expected_values(
    df: pd.DataFrame,
    p_long_col: str = "prob_long",
    p_short_col: str = "prob_short",
    tp_long_col: str = "tp_long_pct",
    sl_long_col: str = "sl_long_pct",
    tp_short_col: str = "tp_short_pct",
    sl_short_col: str = "sl_short_pct",
    roundtrip_cost_pct: float = 0.0022,
) -> pd.DataFrame:
    """
    Calcula EV bruto y neto por lado.

    EV long  = p * TP - (1-p) * SL - coste
    EV short = p * TP - (1-p) * SL - coste

    Todo en porcentaje/retorno decimal.
    """
    out = df.copy()

    p_long = out[p_long_col].fillna(0.0).clip(0.0, 1.0)
    p_short = out[p_short_col].fillna(0.0).clip(0.0, 1.0)

    tp_long = out[tp_long_col].fillna(0.0).clip(lower=0.0)
    sl_long = out[sl_long_col].fillna(0.0).clip(lower=0.0)

    tp_short = out[tp_short_col].fillna(0.0).clip(lower=0.0)
    sl_short = out[sl_short_col].fillna(0.0).clip(lower=0.0)

    out["ev_long_gross"] = p_long * tp_long - (1.0 - p_long) * sl_long
    out["ev_short_gross"] = p_short * tp_short - (1.0 - p_short) * sl_short

    out["ev_long_net"] = out["ev_long_gross"] - roundtrip_cost_pct
    out["ev_short_net"] = out["ev_short_gross"] - roundtrip_cost_pct

    return out


def select_ev_trades(
    df: pd.DataFrame,
    candidate_long_col: str = "candidate_long",
    candidate_short_col: str = "candidate_short",
    min_ev_net: float = 0.0002,
    top_percent_long: float = 0.30,
    top_percent_short: float = 0.30,
) -> pd.DataFrame:
    """
    Selecciona operaciones por EV neto:
    - exige EV neto mínimo
    - dentro de candidatos, opera solo el top X% por EV
    """
    out = df.copy()

    long_mask = (out[candidate_long_col] == 1) & (out["ev_long_net"] > min_ev_net)
    short_mask = (out[candidate_short_col] == 1) & (out["ev_short_net"] > min_ev_net)

    out["selected_long_ev"] = 0
    out["selected_short_ev"] = 0

    if long_mask.any():
        thr_long = out.loc[long_mask, "ev_long_net"].quantile(max(0.0, 1.0 - top_percent_long))
        out.loc[long_mask & (out["ev_long_net"] >= thr_long), "selected_long_ev"] = 1

    if short_mask.any():
        thr_short = out.loc[short_mask, "ev_short_net"].quantile(max(0.0, 1.0 - top_percent_short))
        out.loc[short_mask & (out["ev_short_net"] >= thr_short), "selected_short_ev"] = 1

    return out
