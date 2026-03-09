from pathlib import Path
import sys
import importlib.util
import json
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# ---------------------------------------------------------------------
# Carga dinámica del script base
# ---------------------------------------------------------------------
v6_path = root_dir / "scripts" / "10_walkforward_meta_model_v6.py"

spec = importlib.util.spec_from_file_location("walkforward_v6", v6_path)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo cargar el módulo desde {v6_path}")

walkforward_v6 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = walkforward_v6
spec.loader.exec_module(walkforward_v6)

CONFIG = walkforward_v6.CONFIG
main = walkforward_v6.main

OUT_DIR = root_dir / "artifacts" / "reports" / "walkforward_dual_refine"
walkforward_v6.OUT_DIR = OUT_DIR
walkforward_v6.OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Modos
# baseline congelado + tres variantes dual
# ---------------------------------------------------------------------
CONFIG["modes"] = [
    "short_only_baseline",
    "dual_long_trendup_hard_ref",
    "dual_long_trendup_elite",
    "dual_long_trendup_elite_small",
]

# ---------------------------------------------------------------------
# Attribution sin FutureWarning
# ---------------------------------------------------------------------
def build_oos_attribution_no_warning(trades_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if len(trades_df) == 0:
        empty = pd.DataFrame()
        return {
            "by_direction": empty,
            "by_regime": empty,
            "by_ev_bucket": empty,
            "by_score_bucket": empty,
        }

    t = trades_df.copy()

    t["ev_used"] = np.where(t["direction"] == "long", t["ev_long_regime"], t["ev_short_regime"])
    t["score_used"] = np.where(
        t["direction"] == "long",
        t["setup_score_long_regime"],
        t["setup_score_short_regime"],
    )

    t["ev_bucket"] = pd.cut(
        t["ev_used"],
        bins=[-1.0, 0.004, 0.006, 0.008, 1.0],
        labels=["<=0.004", "(0.004,0.006]", "(0.006,0.008]", ">0.008"],
        include_lowest=True,
        right=True,
    )
    t["score_bucket"] = pd.cut(
        t["score_used"],
        bins=[-1.0, 1.5, 2.0, 3.0, 99.0],
        labels=["<=1.5", "(1.5,2.0]", "(2.0,3.0]", ">3.0"],
        include_lowest=True,
        right=True,
    )

    def summarize(group_col: str) -> pd.DataFrame:
        g = t.groupby(group_col, dropna=False, observed=False).agg(
            n_trades=("net_ret", "size"),
            total_net_ret=("net_ret", "sum"),
            avg_trade_ret=("net_ret", "mean"),
            win_rate=("net_ret", lambda x: float((x > 0).mean())),
            avg_size=("size", "mean"),
        ).reset_index()
        return g.sort_values("total_net_ret", ascending=False).reset_index(drop=True)

    return {
        "by_direction": summarize("direction"),
        "by_regime": summarize("regime_label"),
        "by_ev_bucket": summarize("ev_bucket"),
        "by_score_bucket": summarize("score_bucket"),
    }

walkforward_v6.build_oos_attribution = build_oos_attribution_no_warning

# ---------------------------------------------------------------------
# Long gate: solo tocamos el embudo LONG
# El short queda intacto.
# ---------------------------------------------------------------------
def evaluate_long_pass(row, mode):
    regime = str(row["regime_label"])

    p_long = float(row["prob_long"])
    p_short = float(row["prob_short"])
    ev_long = float(row["ev_long_regime"])
    score_long = float(row["setup_score_long_regime"])
    selected = int(row["selected_long_regime"]) == 1

    if mode == "short_only_baseline":
        return False, {"reason": "mode_block"}

    if regime != "trend_up":
        return False, {"reason": "regime_block", "regime": regime}

    # Referencia: mejor dual actual
    if mode == "dual_long_trendup_hard_ref":
        passed = (
            selected and
            p_long >= 0.54 and
            (p_long - p_short) >= 0.04 and
            ev_long >= 0.0045 and
            score_long >= 1.70
        )
        return passed, {
            "regime": regime,
            "threshold": 0.54,
            "prob_margin": 0.04,
            "min_ev": 0.0045,
            "min_setup": 1.70,
            "selected_flag": int(selected),
        }

    # Más exigente: menos longs, más calidad
    if mode == "dual_long_trendup_elite":
        passed = (
            selected and
            p_long >= 0.555 and
            (p_long - p_short) >= 0.05 and
            ev_long >= 0.0055 and
            score_long >= 1.90
        )
        return passed, {
            "regime": regime,
            "threshold": 0.555,
            "prob_margin": 0.05,
            "min_ev": 0.0055,
            "min_setup": 1.90,
            "selected_flag": int(selected),
        }

    # Igual que elite, pero luego reduciremos tamaño long
    if mode == "dual_long_trendup_elite_small":
        passed = (
            selected and
            p_long >= 0.555 and
            (p_long - p_short) >= 0.05 and
            ev_long >= 0.0055 and
            score_long >= 1.90
        )
        return passed, {
            "regime": regime,
            "threshold": 0.555,
            "prob_margin": 0.05,
            "min_ev": 0.0055,
            "min_setup": 1.90,
            "selected_flag": int(selected),
        }

    return False, {"reason": "unknown_mode", "mode": mode}

walkforward_v6.evaluate_long_pass = evaluate_long_pass

# ---------------------------------------------------------------------
# Conflicto dual: penalizar más al long en variantes elite
# Si ambos pasan, el long solo debe ganar cuando sea claramente superior.
# ---------------------------------------------------------------------
_original_dual_conflict_rank = walkforward_v6.dual_conflict_rank

def dual_conflict_rank(direction: str, row: pd.Series) -> float:
    base_rank = _original_dual_conflict_rank(direction, row)

    if direction == "long":
        p_long = float(row["prob_long"])
        p_short = float(row["prob_short"])
        ev_long = float(row["ev_long_regime"])
        ev_short = float(row["ev_short_regime"])
        score_long = float(row["setup_score_long_regime"])

        penalty = 0.0

        # penalización base al long si el short es competitivo
        if p_short >= 0.53:
            penalty += 0.0015
        if ev_short >= 0.0045:
            penalty += 0.0015
        if (p_long - p_short) < 0.05:
            penalty += 0.0010

        # si el long no es "elite", no debe ganar el desempate fácilmente
        if ev_long < 0.0060:
            penalty += 0.0010
        if score_long < 2.0:
            penalty += 0.0010

        return base_rank - penalty

    return base_rank

walkforward_v6.dual_conflict_rank = dual_conflict_rank

# ---------------------------------------------------------------------
# Tamaño: reducir el presupuesto de riesgo del long en elite_small
# No tocamos el tamaño short.
# ---------------------------------------------------------------------
_original_compute_side_size = walkforward_v6.compute_side_size

def compute_side_size(direction: str, prob: float, threshold: float, setup_score: float, ev_used: float) -> float:
    size = _original_compute_side_size(direction, prob, threshold, setup_score, ev_used)

    # Solo rebajamos el largo cuando es el modo "elite_small".
    # El script base no pasa "mode" aquí, así que hacemos un ajuste estructural:
    # si el largo viene muy apretado (threshold alto + score alto), lo limitamos
    # a un tamaño máximo menor para no estropear el short winner.
    if direction == "long":
        if threshold >= 0.555 and setup_score >= 1.90:
            size = min(size, 0.16)

    return float(size)

walkforward_v6.compute_side_size = compute_side_size

# ---------------------------------------------------------------------
# Ejecutar walk-forward
# ---------------------------------------------------------------------
main()

# ---------------------------------------------------------------------
# Resumen de actividad / seguridad operativa
# Lección de los bots viejos: nada pasa a live sin actividad + reasonability.
# ---------------------------------------------------------------------
def max_consecutive_zeros(values):
    best = 0
    cur = 0
    for v in values:
        if int(v) == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

activity_rows = []

for mode in CONFIG["modes"]:
    mode_dir = OUT_DIR / mode
    folds_path = mode_dir / "folds.csv"
    trades_path = mode_dir / "trades.csv"
    metrics_path = mode_dir / "metrics.json"

    if not folds_path.exists():
        continue

    folds_df = pd.read_csv(folds_path)
    trades_df = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    n_folds = int(metrics["n_folds"])
    n_trades = int(metrics["n_trades"])
    folds_with_trade = int((folds_df["n_trades"] > 0).sum()) if "n_trades" in folds_df.columns else 0
    zero_trade_folds = int((folds_df["n_trades"] == 0).sum()) if "n_trades" in folds_df.columns else n_folds
    max_zero_trade_streak = max_consecutive_zeros(folds_df["n_trades"].tolist()) if "n_trades" in folds_df.columns else n_folds
    active_fold_ratio = float(folds_with_trade / n_folds) if n_folds > 0 else 0.0
    avg_trades_per_active_fold = float(n_trades / max(folds_with_trade, 1))

    avg_hours_between_trades = np.nan
    max_hours_between_trades = np.nan

    if len(trades_df) >= 2 and "timestamp_entry" in trades_df.columns:
        ts = pd.to_datetime(trades_df["timestamp_entry"]).sort_values().reset_index(drop=True)
        diffs_hours = ts.diff().dropna().dt.total_seconds() / 3600.0
        if len(diffs_hours) > 0:
            avg_hours_between_trades = float(diffs_hours.mean())
            max_hours_between_trades = float(diffs_hours.max())

    passes_activity_gate = (
        n_trades >= 20 and
        active_fold_ratio >= 0.67 and
        max_zero_trade_streak <= 2
    )

    activity_rows.append(
        {
            "mode": mode,
            "n_folds": n_folds,
            "n_trades": n_trades,
            "folds_with_trade": folds_with_trade,
            "zero_trade_folds": zero_trade_folds,
            "max_zero_trade_streak": max_zero_trade_streak,
            "active_fold_ratio": active_fold_ratio,
            "avg_trades_per_active_fold": avg_trades_per_active_fold,
            "avg_hours_between_trades": avg_hours_between_trades,
            "max_hours_between_trades": max_hours_between_trades,
            "passes_activity_gate": passes_activity_gate,
            "total_return": float(metrics["total_return"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "sharpe_ratio": float(metrics["sharpe_ratio"]),
            "avg_trade_return": float(metrics["avg_trade_return"]),
            "win_rate_trade": float(metrics["win_rate_trade"]),
        }
    )

activity_df = pd.DataFrame(activity_rows)
activity_df.to_csv(OUT_DIR / "activity_summary.csv", index=False)

print("\n" + "=" * 120)
print("ACTIVITY SUMMARY")
print("=" * 120)
print(activity_df.to_string(index=False))

print("\n" + "=" * 120)
print("DUAL REFINEMENT GOAL")
print("=" * 120)
print(
    "Buscamos que alguna variante dual iguale o supere al short_only_baseline en retorno/Sharpe\n"
    "sin empeorar demasiado drawdown y manteniendo activity gate = True.\n"
)
