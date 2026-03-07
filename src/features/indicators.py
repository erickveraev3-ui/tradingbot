"""
Motor de Indicadores Técnicos.
Calcula: EMA, RSI, MACD, ADX, Bollinger Bands, ATR, Volumen.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from loguru import logger


class TechnicalIndicators:
    """
    Calculadora de indicadores técnicos.
    Todos los métodos son estáticos para facilitar su uso.
    """
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        RSI > 70 = sobrecomprado
        RSI < 30 = sobrevendido
        """
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Returns:
            macd_line: Línea MACD
            signal_line: Línea de señal
            histogram: Histograma (macd - signal)
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Returns:
            upper: Banda superior
            middle: Media (SMA)
            lower: Banda inferior
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range (volatilidad).
        Útil para calcular stop loss dinámico.
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (fuerza de tendencia).
        ADX > 25 = tendencia fuerte
        ADX < 20 = mercado lateral
        
        Returns:
            adx: Índice ADX
            di_plus: +DI (movimiento direccional positivo)
            di_minus: -DI (movimiento direccional negativo)
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Smoothed values
        atr = true_range.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Media móvil de volumen."""
        return volume.rolling(window=period).mean()
    
    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Ratio de volumen actual vs media."""
        vol_sma = volume.rolling(window=period).mean()
        return volume / vol_sma


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos para un DataFrame.
    
    Args:
        df: DataFrame con columnas: timestamp, open, high, low, close, volume
        
    Returns:
        DataFrame con todos los indicadores añadidos
    """
    logger.info("📊 Calculando indicadores técnicos...")
    
    df = df.copy()
    ti = TechnicalIndicators()
    
    # --- EMAs ---
    df['ema_21'] = ti.ema(df['close'], 21)
    df['ema_55'] = ti.ema(df['close'], 55)
    df['ema_200'] = ti.ema(df['close'], 200)
    
    # --- RSI ---
    df['rsi'] = ti.rsi(df['close'], 14)
    
    # --- MACD ---
    df['macd'], df['macd_signal'], df['macd_hist'] = ti.macd(df['close'])
    
    # --- Bollinger Bands ---
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = ti.bollinger_bands(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # --- ATR (volatilidad) ---
    df['atr'] = ti.atr(df['high'], df['low'], df['close'])
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR como % del precio
    
    # --- ADX (fuerza de tendencia) ---
    df['adx'], df['di_plus'], df['di_minus'] = ti.adx(df['high'], df['low'], df['close'])
    
    # --- Volumen ---
    df['vol_sma'] = ti.volume_sma(df['volume'])
    df['vol_ratio'] = ti.volume_ratio(df['volume'])
    
    # --- Señales derivadas ---
    # Tendencia EMA
    df['trend_ema'] = np.where(df['ema_21'] > df['ema_55'], 1, -1)
    df['trend_long'] = np.where(df['close'] > df['ema_200'], 1, -1)
    
    # Cruce MACD
    df['macd_cross'] = np.where(
        (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
        np.where(
            (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0
        )
    )
    
    # RSI zonas
    df['rsi_zone'] = np.where(df['rsi'] > 70, -1,  # Sobrecomprado
                     np.where(df['rsi'] < 30, 1,   # Sobrevendido
                     0))                            # Neutral
    
    # Fuerza de tendencia
    df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)  # 1 = tendencia fuerte
    
    # Retornos
    df['return_1'] = df['close'].pct_change(1)
    df['return_4'] = df['close'].pct_change(4)
    df['return_24'] = df['close'].pct_change(24)
    
    # Limpiar NaN iniciales
    df = df.dropna().reset_index(drop=True)
    
    logger.info(f"✅ Indicadores calculados. Shape: {df.shape}")
    
    return df


def test_indicators():
    """Prueba el cálculo de indicadores."""
    from pathlib import Path
    
    # Cargar datos
    data_path = Path("data/raw/btcusdt_1h.csv")
    
    if not data_path.exists():
        print("❌ Primero ejecuta: python scripts/01_download_data.py")
        return
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"📊 Datos cargados: {len(df)} filas")
    
    # Calcular indicadores
    df_ind = calculate_all_indicators(df)
    
    print(f"\n📈 Columnas después de indicadores:")
    print(df_ind.columns.tolist())
    
    print(f"\n📊 Últimas 5 filas:")
    cols_show = ['timestamp', 'close', 'ema_21', 'rsi', 'macd', 'adx', 'trend_ema']
    print(df_ind[cols_show].tail().to_string())
    
    print(f"\n📊 Estadísticas RSI:")
    print(f"   Min: {df_ind['rsi'].min():.1f}")
    print(f"   Max: {df_ind['rsi'].max():.1f}")
    print(f"   Mean: {df_ind['rsi'].mean():.1f}")


if __name__ == "__main__":
    test_indicators()
