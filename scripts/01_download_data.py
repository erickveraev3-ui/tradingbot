"""
Script 01: Descarga de datos históricos de Binance.
Descarga velas 1H y 4H de BTC/USDT para entrenamiento.

Uso:
    python scripts/01_download_data.py
"""

import sys
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.binance_client import BinanceDataClient
from loguru import logger


def download_historical_data(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: str = "1 Jan 2024",
    output_dir: str = "data/raw"
):
    """
    Descarga datos históricos y los guarda en CSV.
    """
    logger.info("=" * 60)
    logger.info(f"DESCARGA: {symbol} {interval}")
    logger.info("=" * 60)
    
    # Crear directorio si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Inicializar cliente
    client = BinanceDataClient()
    
    # Descargar datos
    logger.info(f"📥 Descargando {symbol} {interval} desde {start_date}...")
    
    df = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=None
    )
    
    # Información del dataset
    logger.info(f"📊 Dataset descargado:")
    logger.info(f"   - Filas: {len(df):,}")
    logger.info(f"   - Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # Estadísticas básicas
    logger.info(f"📈 Estadísticas de precio:")
    logger.info(f"   - Min: ${df['close'].min():,.2f}")
    logger.info(f"   - Max: ${df['close'].max():,.2f}")
    logger.info(f"   - Actual: ${df['close'].iloc[-1]:,.2f}")
    
    # Guardar CSV
    filename = f"{symbol.lower()}_{interval}.csv"
    filepath = Path(output_dir) / filename
    
    df.to_csv(filepath, index=False)
    logger.info(f"💾 Guardado en: {filepath}")
    
    return df


def main():
    """Función principal."""
    
    print("\n" + "=" * 60)
    print("       DESCARGA DE DATOS HISTÓRICOS BTC")
    print("=" * 60 + "\n")
    
    # Descargar velas 1H desde 2020 (5 años de datos)
    df_1h = download_historical_data(
        symbol="BTCUSDT",
        interval="1h",
        start_date="1 Jan 2020",  # 5 años de datos
        output_dir="data/raw"
    )
    
    print("\n")
    
    # Descargar velas 4H
    df_4h = download_historical_data(
        symbol="BTCUSDT",
        interval="4h",
        start_date="1 Jan 2020",
        output_dir="data/raw"
    )
    
    print("\n")
    
    # Descargar velas 1D
    df_1d = download_historical_data(
        symbol="BTCUSDT",
        interval="1d",
        start_date="1 Jan 2019",  # 6 años para diario
        output_dir="data/raw"
    )
    
    # Resumen final
    print("\n" + "=" * 60)
    print("✅ DESCARGA COMPLETADA")
    print("=" * 60)
    print(f"\nArchivos creados en data/raw/:")
    print(f"  • btcusdt_1h.csv  ({len(df_1h):,} velas)")
    print(f"  • btcusdt_4h.csv  ({len(df_4h):,} velas)")
    print(f"  • btcusdt_1d.csv  ({len(df_1d):,} velas)")
    
    print("\n📋 Primeras 3 filas de 1H:")
    print(df_1h.head(3).to_string())
    
    print("\n📋 Últimas 3 filas de 1H:")
    print(df_1h.tail(3).to_string())


if __name__ == "__main__":
    main()