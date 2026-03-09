def build_candidate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    candidate_v2_soft:
    - algo más permisiva que la versión actual
    - sigue siendo estructurada
    - deja el trabajo duro al meta-model + EV + ranking
    """
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

    structure_score_long = df["structure_score_long"] if "structure_score_long" in df.columns else pd.Series(0.0, index=df.index)
    structure_score_short = df["structure_score_short"] if "structure_score_short" in df.columns else pd.Series(0.0, index=df.index)

    break_swing_high = df["break_swing_high"] if "break_swing_high" in df.columns else pd.Series(0.0, index=df.index)
    break_swing_low = df["break_swing_low"] if "break_swing_low" in df.columns else pd.Series(0.0, index=df.index)

    # ------------------------------------------------------------------
    # Tendencia algo más flexible
    # ------------------------------------------------------------------
    trend_long = (ema_trend > 0) & (adx >= 0.12)
    trend_short = (ema_trend < 0) & (adx >= 0.12)

    trend_candidate_long = (pattern_long_score >= 0.70) & trend_long
    trend_candidate_short = (pattern_short_score >= 0.70) & trend_short

    # ------------------------------------------------------------------
    # Breakout / squeeze con soporte estructural
    # ------------------------------------------------------------------
    breakout_candidate_long = (
        (pattern_long_score >= 0.55) &
        ((squeeze_up > 0) | (break_swing_high > 0)) &
        (structure_score_long >= 0.35)
    )

    breakout_candidate_short = (
        (pattern_short_score >= 0.55) &
        ((squeeze_down > 0) | (break_swing_low > 0)) &
        (structure_score_short >= 0.35)
    )

    # ------------------------------------------------------------------
    # Reversión estructural
    # ------------------------------------------------------------------
    reversal_candidate_long = (
        (pattern_long_score >= 0.55) &
        ((db_confirm > 0) | (sweep_long > 0)) &
        (structure_score_long >= 0.45)
    )

    reversal_candidate_short = (
        (pattern_short_score >= 0.55) &
        ((dt_confirm > 0) | (sweep_short > 0)) &
        (structure_score_short >= 0.45)
    )

    # ------------------------------------------------------------------
    # Combo score algo más abierto
    # ------------------------------------------------------------------
    combo_long_score = (
        pattern_long_score
        + 0.35 * (squeeze_up > 0).astype(float)
        + 0.30 * (db_confirm > 0).astype(float)
        + 0.25 * (sweep_long > 0).astype(float)
        + 0.25 * (break_swing_high > 0).astype(float)
        + 0.20 * structure_score_long.clip(lower=0.0)
        + 0.15 * trend_long.astype(float)
    )

    combo_short_score = (
        pattern_short_score
        + 0.35 * (squeeze_down > 0).astype(float)
        + 0.30 * (dt_confirm > 0).astype(float)
        + 0.25 * (sweep_short > 0).astype(float)
        + 0.25 * (break_swing_low > 0).astype(float)
        + 0.20 * structure_score_short.clip(lower=0.0)
        + 0.15 * trend_short.astype(float)
    )

    combo_candidate_long = combo_long_score >= 1.00
    combo_candidate_short = combo_short_score >= 1.00

    candidate_long = (
        trend_candidate_long
        | breakout_candidate_long
        | reversal_candidate_long
        | combo_candidate_long
    ).astype(int)

    candidate_short = (
        trend_candidate_short
        | breakout_candidate_short
        | reversal_candidate_short
        | combo_candidate_short
    ).astype(int)

    df["candidate_long"] = candidate_long
    df["candidate_short"] = candidate_short
    df["combo_long_score"] = combo_long_score
    df["combo_short_score"] = combo_short_score

    return df