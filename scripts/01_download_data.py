import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from loguru import logger
from src.data.binance_client import BinanceDataClient


RAW_DIR = root_dir / "data" / "raw"


def download_symbol_interval(
    client: BinanceDataClient,
    symbol: str,
    interval: str,
    start_date: str,
    output_dir: Path,
) -> None:
    """
    Descarga histórico de un símbolo/intervalo y lo guarda en CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"DESCARGA {symbol} {interval}")
    logger.info("=" * 70)

    df = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=None,
    )

    if df.empty:
        raise ValueError(f"Descarga vacía para {symbol} {interval}")

    filename = f"{symbol.lower()}_{interval}.csv"
    path = output_dir / filename
    df.to_csv(path, index=False)

    logger.info(f"✅ Guardado: {path}")
    logger.info(f"   Filas: {len(df):,}")
    logger.info(f"   Rango: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    logger.info(f"   Último close: {df['close'].iloc[-1]:.4f}")


def main():
    print("\n" + "=" * 70)
    print("DESCARGA DE DATOS HISTÓRICOS - BTC / ETH / SOL")
    print("=" * 70 + "\n")

    client = BinanceDataClient()

    # =========================
    # DATOS 1H
    # =========================
    symbols_1h = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    for symbol in symbols_1h:
        download_symbol_interval(
            client=client,
            symbol=symbol,
            interval="1h",
            start_date="1 Jan 2020",
            output_dir=RAW_DIR,
        )

    # =========================
    # DATOS 15M
    # =========================
    symbols_15m = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    for symbol in symbols_15m:
        download_symbol_interval(
            client=client,
            symbol=symbol,
            interval="15m",
            start_date="1 Jan 2020",
            output_dir=RAW_DIR,
        )

    # =========================
    # BTC 4H opcional
    # =========================
    download_symbol_interval(
        client=client,
        symbol="BTCUSDT",
        interval="4h",
        start_date="1 Jan 2020",
        output_dir=RAW_DIR,
    )

    print("\n" + "=" * 70)
    print("✅ DESCARGA COMPLETADA")
    print("=" * 70)
    print("\nArchivos esperados en data/raw/:")
    print("  - btcusdt_1h.csv")
    print("  - ethusdt_1h.csv")
    print("  - solusdt_1h.csv")
    print("  - btcusdt_15m.csv")
    print("  - ethusdt_15m.csv")
    print("  - solusdt_15m.csv")
    print("  - btcusdt_4h.csv")


if __name__ == "__main__":
    main()