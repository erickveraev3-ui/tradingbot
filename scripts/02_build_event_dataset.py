from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
from loguru import logger

from src.features.feature_builder import FeatureBuilder, load_csv
from src.features.structure_features import add_structure_features


RAW_DIR = root_dir / "data" / "raw"
PROCESSED_DIR = root_dir / "data" / "processed"


def validate_dataframe(df: pd.DataFrame, name: str):
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{name}: falta columna {col}")

    if df.empty:
        raise ValueError(f"{name}: dataframe vacío")

    if df["timestamp"].duplicated().any():
        raise ValueError(f"{name}: timestamps duplicados")

    df.sort_values("timestamp", inplace=True)


def align_dataframes(btc, eth, sol):
    common_ts = (
        set(btc["timestamp"])
        .intersection(set(eth["timestamp"]))
        .intersection(set(sol["timestamp"]))
    )
    common_ts = sorted(common_ts)

    btc = btc[btc["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    eth = eth[eth["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    sol = sol[sol["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    return btc, eth, sol


def build_event_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Targets de eventos operables.
    No buscamos retorno exacto, sino si aparece una oportunidad explotable.
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr = df["atr"] if "atr" in df.columns else (high - low).rolling(14).mean()
    atr = atr.replace(0, np.nan).ffill().bfill()

    # =========================
    # Horizontes futuros
    # =========================
    future_high_4 = high.shift(-1).rolling(4).max()
    future_low_4 = low.shift(-1).rolling(4).min()

    future_high_8 = high.shift(-1).rolling(8).max()
    future_low_8 = low.shift(-1).rolling(8).min()

    # Move potencial relativo al close actual
    up_move_4 = (future_high_4 - close) / close
    down_move_4 = (close - future_low_4) / close

    up_move_8 = (future_high_8 - close) / close
    down_move_8 = (close - future_low_8) / close

    # Stop adverso simple
    adverse_down_4 = (close - future_low_4) / close
    adverse_up_4 = (future_high_4 - close) / close

    # =========================
    # Eventos principales
    # =========================
    # Tradeable move: hay desplazamiento suficiente y la excursión adversa no es desproporcionada
    df["event_long_4h"] = ((up_move_4 >= 0.0040) & (adverse_down_4 <= 0.0060)).astype(int)
    df["event_short_4h"] = ((down_move_4 >= 0.0040) & (adverse_up_4 <= 0.0060)).astype(int)

    df["event_long_8h"] = ((up_move_8 >= 0.0060) & (down_move_8 <= 0.0080)).astype(int)
    df["event_short_8h"] = ((down_move_8 >= 0.0060) & (up_move_8 <= 0.0080)).astype(int)

    # =========================
    # Eventos estructurales
    # =========================
    # Breakout: precio cerca del máximo y luego rompe
    if "dist_to_high_12_atr" in df.columns:
        near_high = df["dist_to_high_12_atr"] < 0.4
        near_low = df["dist_to_low_12_atr"] < 0.4
    else:
        near_high = pd.Series(False, index=df.index)
        near_low = pd.Series(False, index=df.index)

    df["event_breakout_up"] = (near_high & (up_move_4 >= 0.0040)).astype(int)
    df["event_breakdown_down"] = (near_low & (down_move_4 >= 0.0040)).astype(int)

    # Reversal desde doble suelo / doble techo
    if "double_bottom_proxy" in df.columns:
        df["event_reversal_long"] = ((df["double_bottom_proxy"] > 0) & (up_move_8 >= 0.0060)).astype(int)
    else:
        df["event_reversal_long"] = 0

    if "double_top_proxy" in df.columns:
        df["event_reversal_short"] = ((df["double_top_proxy"] > 0) & (down_move_8 >= 0.0060)).astype(int)
    else:
        df["event_reversal_short"] = 0

    # Label principal para empezar: evento long rentable de corto plazo
    df["target_event_long"] = df["event_long_4h"]
    df["target_event_short"] = df["event_short_4h"]

    # Contexto informativo adicional
    df["future_up_move_4h"] = up_move_4
    df["future_down_move_4h"] = down_move_4
    df["future_up_move_8h"] = up_move_8
    df["future_down_move_8h"] = down_move_8

    df = df.dropna().reset_index(drop=True)
    return df


def main():
    logger.info("=" * 70)
    logger.info("CREANDO DATASET DE EVENTOS ESTRUCTURALES")
    logger.info("=" * 70)

    btc_path = RAW_DIR / "btcusdt_1h.csv"
    eth_path = RAW_DIR / "ethusdt_1h.csv"
    sol_path = RAW_DIR / "solusdt_1h.csv"

    for p in [btc_path, eth_path, sol_path]:
        if not p.exists():
            raise FileNotFoundError(f"No existe {p}")

    btc = load_csv(str(btc_path))
    eth = load_csv(str(eth_path))
    sol = load_csv(str(sol_path))

    validate_dataframe(btc, "BTC")
    validate_dataframe(eth, "ETH")
    validate_dataframe(sol, "SOL")

    btc, eth, sol = align_dataframes(btc, eth, sol)

    logger.info(f"Filas alineadas: {len(btc):,}")

    # Features base + contexto ETH/SOL
    builder = FeatureBuilder()
    df = builder.build(btc, eth, sol)

    # Añadir estructura profesional
    df = add_structure_features(df)

    # Añadir targets de eventos
    df = build_event_targets(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset_btc_event_1h.csv"
    df.to_csv(out_path, index=False)

    logger.info(f"Dataset guardado en: {out_path}")
    logger.info(f"Filas finales: {len(df):,}")
    logger.info(f"Columnas: {len(df.columns)}")

    # Resumen de eventos
    event_cols = [
        "target_event_long",
        "target_event_short",
        "event_breakout_up",
        "event_breakdown_down",
        "event_reversal_long",
        "event_reversal_short",
    ]

    logger.info("Frecuencia de eventos:")
    for col in event_cols:
        if col in df.columns:
            logger.info(f"{col}: {df[col].mean():.2%}")


if __name__ == "__main__":
    main()
