from pathlib import Path
import sys
import importlib.util

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# ---------------------------------------------------------------------
# Carga dinámica correcta del script base
# ---------------------------------------------------------------------
v6_path = root_dir / "scripts" / "10_walkforward_meta_model_v6.py"

spec = importlib.util.spec_from_file_location("walkforward_v6", v6_path)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo cargar el módulo desde {v6_path}")

walkforward_v6 = importlib.util.module_from_spec(spec)

# CRÍTICO en Python 3.12 para dataclass y resolución de módulo
sys.modules[spec.name] = walkforward_v6

spec.loader.exec_module(walkforward_v6)

# ---------------------------------------------------------------------
# Alias locales
# ---------------------------------------------------------------------
CONFIG = walkforward_v6.CONFIG
main = walkforward_v6.main

# carpeta de salida separada para no mezclar resultados
walkforward_v6.OUT_DIR = root_dir / "artifacts" / "reports" / "walkforward_dual_long_research"
walkforward_v6.OUT_DIR.mkdir(parents=True, exist_ok=True)

# modos del experimento
CONFIG["modes"] = [
    "short_only_baseline",
    "dual_long_trendup_only",
    "dual_long_trendup_strict",
]

# ---------------------------------------------------------------------
# Override quirúrgico: solo tocar el embudo LONG
# El short queda exactamente igual que en el script base.
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

    # ------------------------------------------------------------
    # Variante 1: dual con long solo en trend_up
    # ------------------------------------------------------------
    if mode == "dual_long_trendup_only":
        if regime != "trend_up":
            return False, {"reason": "regime_block", "regime": regime}

        passed = (
            selected and
            p_long >= 0.52 and
            (p_long - p_short) >= 0.02 and
            ev_long >= 0.0030 and
            score_long >= 1.40
        )

        return passed, {
            "regime": regime,
            "threshold": 0.52,
            "prob_margin": 0.02,
            "min_ev": 0.0030,
            "min_setup": 1.40,
            "selected_flag": int(selected),
        }

    # ------------------------------------------------------------
    # Variante 2: dual con trend_up normal + range muy estricto
    # ------------------------------------------------------------
    if mode == "dual_long_trendup_strict":
        if regime == "trend_up":
            passed = (
                selected and
                p_long >= 0.52 and
                (p_long - p_short) >= 0.02 and
                ev_long >= 0.0030 and
                score_long >= 1.40
            )
            return passed, {
                "regime": regime,
                "threshold": 0.52,
                "prob_margin": 0.02,
                "min_ev": 0.0030,
                "min_setup": 1.40,
                "selected_flag": int(selected),
            }

        if regime == "range":
            passed = (
                selected and
                p_long >= 0.58 and
                (p_long - p_short) >= 0.05 and
                ev_long >= 0.0060 and
                score_long >= 2.00
            )
            return passed, {
                "regime": regime,
                "threshold": 0.58,
                "prob_margin": 0.05,
                "min_ev": 0.0060,
                "min_setup": 2.00,
                "selected_flag": int(selected),
            }

        return False, {"reason": "regime_block", "regime": regime}

    return False, {"reason": "unknown_mode", "mode": mode}


# inyectar override
walkforward_v6.evaluate_long_pass = evaluate_long_pass

# ejecutar
main()
