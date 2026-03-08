import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import pandas as pd
from loguru import logger

from src.features.feature_builder import FeatureBuilder, load_csv


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

    btc = btc[btc["timestamp"].isin(common_ts)]
    eth = eth[eth["timestamp"].isin(common_ts)]
    sol = sol[sol["timestamp"].isin(common_ts)]

    btc = btc.sort_values("timestamp").reset_index(drop=True)
    eth = eth.sort_values("timestamp").reset_index(drop=True)
    sol = sol.sort_values("timestamp").reset_index(drop=True)

    return btc, eth, sol


def main():

    logger.info("=" * 60)
    logger.info("CREANDO DATASET DE FEATURES")
    logger.info("=" * 60)

    btc_path = RAW_DIR / "btcusdt_1h.csv"
    eth_path = RAW_DIR / "ethusdt_1h.csv"
    sol_path = RAW_DIR / "solusdt_1h.csv"

    if not btc_path.exists():
        raise FileNotFoundError("No existe btcusdt_1h.csv")

    if not eth_path.exists():
        raise FileNotFoundError("No existe ethusdt_1h.csv")

    if not sol_path.exists():
        raise FileNotFoundError("No existe solusdt_1h.csv")

    logger.info("Cargando datos...")

    btc = load_csv(str(btc_path))
    eth = load_csv(str(eth_path))
    sol = load_csv(str(sol_path))

    validate_dataframe(btc, "BTC")
    validate_dataframe(eth, "ETH")
    validate_dataframe(sol, "SOL")

    logger.info(f"BTC filas: {len(btc):,}")
    logger.info(f"ETH filas: {len(eth):,}")
    logger.info(f"SOL filas: {len(sol):,}")

    logger.info("Alineando timestamps...")

    btc, eth, sol = align_dataframes(btc, eth, sol)

    logger.info(f"Filas tras alineación: {len(btc):,}")

    if len(btc) < 1000:
        raise ValueError("Dataset demasiado pequeño tras alineación")

    logger.info("Construyendo features...")

    builder = FeatureBuilder()
    df = builder.build(btc, eth, sol)

    logger.info(f"Dataset final filas: {len(df):,}")
    logger.info(f"Columnas: {len(df.columns)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    out_path = PROCESSED_DIR / "dataset_btc_context_1h.csv"
    df.to_csv(out_path, index=False)

    logger.info(f"Dataset guardado en: {out_path}")

    logger.info("Resumen targets:")

    logger.info(
        f"target_direction positivos: {df['target_direction'].mean():.2%}"
    )

    logger.info(
        f"retorno medio 1h: {df['target_return_1h'].mean():.6f}"
    )

    logger.info(
        f"retorno medio 4h: {df['target_return_4h'].mean():.6f}"
    )


if __name__ == "__main__":
    main()
