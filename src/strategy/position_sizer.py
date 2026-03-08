from __future__ import annotations

import math


def dynamic_position_size(
    prob: float,
    threshold: float,
    setup_score: float,
    base_size: float = 0.20,
    max_size: float = 1.00,
    prob_scale: float = 2.0,
    score_scale: float = 0.25,
) -> float:
    """
    Tamaño dinámico simple y robusto.

    - parte de un tamaño base
    - crece si la probabilidad supera el threshold
    - crece un poco si el setup_score es fuerte
    """
    excess_prob = max(0.0, prob - threshold)

    size = base_size
    size += prob_scale * excess_prob
    size += score_scale * max(0.0, setup_score - 0.5)

    size = max(0.0, min(size, max_size))
    return float(size)
