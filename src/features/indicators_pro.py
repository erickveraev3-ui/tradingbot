"""
Indicadores Profesionales de Grado Institucional.

Incluye:
- VWAP (Volume Weighted Average Price)
- CVD (Cumulative Volume Delta)
- OBV (On Balance Volume)
- Fibonacci Dinámico
- Market Structure (HH, HL, LH, LL)
- Liquidity Zones
- Order Blocks
- Fair Value Gaps
- Divergencias RSI/Price
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from loguru import logger


# =============================================================================
# INDICADORES DE VOLUMEN INSTITUCIONAL
# =============================================================================

def calc_vwap(df: pd.DataFrame, reset_period: str = None) -> pd.Series:
    """
    Volume Weighted Average Price.
    Precio promedio ponderado por volumen - usado por instituciones.
    
    Args:
        df: DataFrame con high, low, close, volume
        reset_period: 'D' para reset diario, None para acumulado
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    if reset_period:
        # Reset por período (típicamente diario)
        df_temp = df.copy()
        df_temp['tp_vol'] = tp * df['volume']
        df_temp['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        vwap = df_temp.groupby('date').apply(
            lambda x: x['tp_vol'].cumsum() / x['volume'].cumsum()
        ).reset_index(level=0, drop=True)
    else:
        # VWAP acumulado (rolling)
        tp_vol_cum = (tp * df['volume']).rolling(50).sum()
        vol_cum = df['volume'].rolling(50).sum()
        vwap = tp_vol_cum / vol_cum
    
    return vwap


def calc_vwap_bands(df: pd.DataFrame, std_mult: List[float] = [1, 2]) -> Dict[str, pd.Series]:
    """
    VWAP con bandas de desviación estándar.
    """
    vwap = calc_vwap(df)
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Rolling std del precio respecto al VWAP
    std = (tp - vwap).rolling(50).std()
    
    bands = {'vwap': vwap}
    for mult in std_mult:
        bands[f'vwap_upper_{mult}'] = vwap + std * mult
        bands[f'vwap_lower_{mult}'] = vwap - std * mult
    
    # Distancia normalizada del precio al VWAP
    bands['vwap_distance'] = (df['close'] - vwap) / std
    
    return bands


def calc_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative Volume Delta (aproximación).
    Estima presión compradora vs vendedora.
    
    Lógica: Si cierra arriba de apertura = volumen comprador
            Si cierra abajo de apertura = volumen vendedor
    """
    # Método 1: Simple
    delta_simple = np.where(
        df['close'] > df['open'], 
        df['volume'],
        np.where(df['close'] < df['open'], -df['volume'], 0)
    )
    
    return pd.Series(np.cumsum(delta_simple), index=df.index)


def calc_cvd_advanced(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    CVD avanzado con múltiples métricas.
    """
    # Proporción de la vela
    candle_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (candle_range + 1e-10)
    
    # Delta ponderado por posición del cierre
    buy_volume = df['volume'] * close_position
    sell_volume = df['volume'] * (1 - close_position)
    delta = buy_volume - sell_volume
    
    cvd = delta.cumsum()
    
    # CVD momentum (cambio del CVD)
    cvd_momentum = cvd.diff(10)
    
    # Divergencia precio vs CVD
    price_change = df['close'].pct_change(20)
    cvd_change = cvd.pct_change(20)
    
    # Si precio sube pero CVD baja = divergencia bajista
    # Si precio baja pero CVD sube = divergencia alcista
    divergence = np.sign(price_change) != np.sign(cvd_change)
    
    return {
        'cvd': cvd,
        'cvd_momentum': cvd_momentum,
        'cvd_divergence': divergence.astype(int),
        'buy_volume_pct': buy_volume / (buy_volume + sell_volume)
    }


def calc_obv(df: pd.DataFrame) -> pd.Series:
    """
    On Balance Volume.
    Acumula volumen según dirección del precio.
    """
    direction = np.sign(df['close'].diff())
    obv = (direction * df['volume']).cumsum()
    
    return obv


def calc_obv_advanced(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    OBV con señales adicionales.
    """
    obv = calc_obv(df)
    
    # OBV EMA para suavizar
    obv_ema = obv.ewm(span=20).mean()
    
    # OBV momentum
    obv_momentum = obv - obv_ema
    
    # Divergencia OBV vs Precio
    price_highs = df['close'].rolling(20).max() == df['close']
    obv_highs = obv.rolling(20).max() == obv
    
    # Divergencia bajista: precio hace high pero OBV no
    bearish_div = price_highs & ~obv_highs
    
    price_lows = df['close'].rolling(20).min() == df['close']
    obv_lows = obv.rolling(20).min() == obv
    
    # Divergencia alcista: precio hace low pero OBV no
    bullish_div = price_lows & ~obv_lows
    
    return {
        'obv': obv,
        'obv_ema': obv_ema,
        'obv_momentum': obv_momentum,
        'obv_bullish_div': bullish_div.astype(int),
        'obv_bearish_div': bearish_div.astype(int)
    }


# =============================================================================
# FIBONACCI DINÁMICO
# =============================================================================

def calc_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> Dict[str, pd.Series]:
    """
    Niveles de Fibonacci dinámicos basados en swing high/low recientes.
    """
    high = df['high'].rolling(lookback).max()
    low = df['low'].rolling(lookback).min()
    diff = high - low
    
    # Niveles de retroceso
    levels = {
        'fib_0': high,                    # 0%
        'fib_236': high - diff * 0.236,   # 23.6%
        'fib_382': high - diff * 0.382,   # 38.2%
        'fib_500': high - diff * 0.500,   # 50%
        'fib_618': high - diff * 0.618,   # 61.8%
        'fib_786': high - diff * 0.786,   # 78.6%
        'fib_100': low,                   # 100%
    }
    
    # Extensiones
    levels['fib_ext_1272'] = high + diff * 0.272  # 127.2%
    levels['fib_ext_1618'] = high + diff * 0.618  # 161.8%
    
    # Distancia del precio actual a cada nivel (normalizada)
    price = df['close']
    distances = {}
    for name, level in levels.items():
        dist = (price - level) / (diff + 1e-10)
        distances[f'{name}_dist'] = dist
    
    # Nivel más cercano
    all_dists = pd.DataFrame({k: np.abs(v) for k, v in distances.items()})
    nearest_level = all_dists.idxmin(axis=1).str.replace('_dist', '')
    
    # Codificar nivel cercano
    level_map = {'fib_0': 0, 'fib_236': 1, 'fib_382': 2, 'fib_500': 3, 
                 'fib_618': 4, 'fib_786': 5, 'fib_100': 6, 
                 'fib_ext_1272': 7, 'fib_ext_1618': 8}
    distances['fib_nearest'] = nearest_level.map(level_map)
    distances['fib_min_dist'] = all_dists.min(axis=1)
    
    return distances


def calc_fibonacci_support_resistance(df: pd.DataFrame, lookback: int = 100) -> Dict[str, pd.Series]:
    """
    Identifica si el precio está en zona de soporte/resistencia Fibonacci.
    """
    fib = calc_fibonacci_levels(df, lookback)
    
    # Zona = precio está dentro del 2% del nivel
    threshold = 0.02
    
    in_zone = (fib['fib_min_dist'].abs() < threshold).astype(int)
    
    # Tipo de zona (soporte si precio > nivel, resistencia si precio < nivel)
    # Simplificado: basado en fib_500_dist
    zone_type = np.where(fib['fib_500_dist'] > 0, 1, -1)  # 1=soporte, -1=resistencia
    
    return {
        'fib_in_zone': in_zone,
        'fib_zone_type': zone_type,
        'fib_500_dist': fib['fib_500_dist'],
        'fib_618_dist': fib['fib_618_dist'],
        'fib_382_dist': fib['fib_382_dist'],
    }


# =============================================================================
# MARKET STRUCTURE (SMC - Smart Money Concepts)
# =============================================================================

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Dict[str, pd.Series]:
    """
    Identifica Swing Highs y Swing Lows.
    """
    highs = df['high'].values
    lows = df['low'].values
    
    swing_high = np.zeros(len(df))
    swing_low = np.zeros(len(df))
    
    for i in range(lookback, len(df) - lookback):
        # Swing High: punto más alto en ventana
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swing_high[i] = highs[i]
        
        # Swing Low: punto más bajo en ventana
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swing_low[i] = lows[i]
    
    return {
        'swing_high': pd.Series(swing_high, index=df.index),
        'swing_low': pd.Series(swing_low, index=df.index)
    }


def calc_market_structure(df: pd.DataFrame, lookback: int = 20) -> Dict[str, pd.Series]:
    """
    Calcula estructura de mercado:
    - HH (Higher High): Máximo más alto que el anterior
    - HL (Higher Low): Mínimo más alto que el anterior
    - LH (Lower High): Máximo más bajo que el anterior
    - LL (Lower Low): Mínimo más bajo que el anterior
    - BOS (Break of Structure)
    - CHoCH (Change of Character)
    """
    swings = find_swing_points(df, lookback=5)
    
    # Últimos swing points válidos
    sh = swings['swing_high'].replace(0, np.nan).ffill()
    sl = swings['swing_low'].replace(0, np.nan).ffill()
    
    # Previous swing points
    sh_prev = sh.shift(lookback)
    sl_prev = sl.shift(lookback)
    
    # Estructura
    hh = (sh > sh_prev).astype(int)  # Higher High
    hl = (sl > sl_prev).astype(int)  # Higher Low
    lh = (sh < sh_prev).astype(int)  # Lower High
    ll = (sl < sl_prev).astype(int)  # Lower Low
    
    # Tendencia basada en estructura
    # Uptrend: HH y HL
    # Downtrend: LH y LL
    bullish_structure = (hh & hl).rolling(5).sum() > 0
    bearish_structure = (lh & ll).rolling(5).sum() > 0
    
    structure_trend = np.where(bullish_structure, 1, 
                              np.where(bearish_structure, -1, 0))
    
    # BOS (Break of Structure)
    # Precio rompe el último swing high (bullish BOS)
    # Precio rompe el último swing low (bearish BOS)
    bullish_bos = (df['close'] > sh.shift(1)) & (df['close'].shift(1) <= sh.shift(1))
    bearish_bos = (df['close'] < sl.shift(1)) & (df['close'].shift(1) >= sl.shift(1))
    
    # CHoCH (Change of Character)
    # Cambio de estructura de bullish a bearish o viceversa
    prev_trend = pd.Series(structure_trend).shift(1)
    choch = (structure_trend != prev_trend) & (prev_trend != 0)
    
    return {
        'ms_hh': hh,
        'ms_hl': hl,
        'ms_lh': lh,
        'ms_ll': ll,
        'ms_trend': pd.Series(structure_trend, index=df.index),
        'ms_bullish_bos': bullish_bos.astype(int),
        'ms_bearish_bos': bearish_bos.astype(int),
        'ms_choch': choch.astype(int)
    }


# =============================================================================
# LIQUIDITY ZONES & ORDER BLOCKS
# =============================================================================

def calc_liquidity_zones(df: pd.DataFrame, lookback: int = 50) -> Dict[str, pd.Series]:
    """
    Identifica zonas de liquidez.
    Alta liquidez = muchas órdenes acumuladas = zonas de reversión potencial.
    """
    # Volume profile simplificado
    vol_mean = df['volume'].rolling(lookback).mean()
    vol_std = df['volume'].rolling(lookback).std()
    
    # Zonas de alto volumen
    vol_zscore = (df['volume'] - vol_mean) / (vol_std + 1e-10)
    high_vol_zone = (vol_zscore > 1.5).astype(int)
    
    # Swing points con alto volumen = liquidez
    swings = find_swing_points(df)
    
    # Liquidez en highs (sell stops)
    liq_above = (swings['swing_high'] > 0) & (vol_zscore > 1)
    
    # Liquidez en lows (buy stops)
    liq_below = (swings['swing_low'] > 0) & (vol_zscore > 1)
    
    # Distancia a última zona de liquidez
    last_liq_high = swings['swing_high'].replace(0, np.nan).ffill()
    last_liq_low = swings['swing_low'].replace(0, np.nan).ffill()
    
    dist_to_liq_high = (last_liq_high - df['close']) / df['close']
    dist_to_liq_low = (df['close'] - last_liq_low) / df['close']
    
    return {
        'liq_vol_zscore': vol_zscore,
        'liq_high_vol_zone': high_vol_zone,
        'liq_above': liq_above.astype(int),
        'liq_below': liq_below.astype(int),
        'liq_dist_high': dist_to_liq_high,
        'liq_dist_low': dist_to_liq_low
    }


def calc_order_blocks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Identifica Order Blocks (zonas institucionales).
    
    Bullish OB: Última vela bajista antes de impulso alcista fuerte
    Bearish OB: Última vela alcista antes de impulso bajista fuerte
    """
    # Detectar impulsos fuertes
    returns = df['close'].pct_change()
    impulse_threshold = returns.rolling(20).std() * 2
    
    bullish_impulse = returns > impulse_threshold
    bearish_impulse = returns < -impulse_threshold
    
    # Vela anterior al impulso
    prev_bearish = df['close'].shift(1) < df['open'].shift(1)  # Vela roja
    prev_bullish = df['close'].shift(1) > df['open'].shift(1)  # Vela verde
    
    # Order Blocks
    bullish_ob = bullish_impulse & prev_bearish  # OB alcista
    bearish_ob = bearish_impulse & prev_bullish  # OB bajista
    
    # Precio del OB
    bullish_ob_price = np.where(bullish_ob, df['low'].shift(1), np.nan)
    bearish_ob_price = np.where(bearish_ob, df['high'].shift(1), np.nan)
    
    # Último OB válido
    last_bullish_ob = pd.Series(bullish_ob_price).ffill()
    last_bearish_ob = pd.Series(bearish_ob_price).ffill()
    
    # Distancia al último OB
    dist_bullish_ob = (df['close'] - last_bullish_ob) / df['close']
    dist_bearish_ob = (last_bearish_ob - df['close']) / df['close']
    
    return {
        'ob_bullish': bullish_ob.astype(int),
        'ob_bearish': bearish_ob.astype(int),
        'ob_dist_bullish': dist_bullish_ob,
        'ob_dist_bearish': dist_bearish_ob,
        'ob_near_bullish': (dist_bullish_ob.abs() < 0.02).astype(int),
        'ob_near_bearish': (dist_bearish_ob.abs() < 0.02).astype(int)
    }


# =============================================================================
# FAIR VALUE GAPS (FVG / IMBALANCE)
# =============================================================================

def calc_fair_value_gaps(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Identifica Fair Value Gaps (imbalances).
    
    Bullish FVG: Low[i] > High[i-2] (gap alcista)
    Bearish FVG: High[i] < Low[i-2] (gap bajista)
    """
    # Bullish FVG
    bullish_fvg = df['low'] > df['high'].shift(2)
    bullish_fvg_size = np.where(bullish_fvg, df['low'] - df['high'].shift(2), 0)
    
    # Bearish FVG
    bearish_fvg = df['high'] < df['low'].shift(2)
    bearish_fvg_size = np.where(bearish_fvg, df['low'].shift(2) - df['high'], 0)
    
    # FVG no rellenados (precio no ha vuelto)
    # Simplificado: si el precio actual está lejos del FVG
    last_bullish_fvg = pd.Series(np.where(bullish_fvg, df['low'], np.nan)).ffill()
    last_bearish_fvg = pd.Series(np.where(bearish_fvg, df['high'], np.nan)).ffill()
    
    fvg_bullish_unfilled = df['close'] > last_bullish_fvg
    fvg_bearish_unfilled = df['close'] < last_bearish_fvg
    
    return {
        'fvg_bullish': bullish_fvg.astype(int),
        'fvg_bearish': bearish_fvg.astype(int),
        'fvg_bullish_size': pd.Series(bullish_fvg_size, index=df.index) / df['close'],
        'fvg_bearish_size': pd.Series(bearish_fvg_size, index=df.index) / df['close'],
        'fvg_bullish_unfilled': fvg_bullish_unfilled.astype(int),
        'fvg_bearish_unfilled': fvg_bearish_unfilled.astype(int)
    }


# =============================================================================
# DIVERGENCIAS
# =============================================================================

def calc_divergences(df: pd.DataFrame, indicator: str = 'rsi', lookback: int = 14) -> Dict[str, pd.Series]:
    """
    Detecta divergencias entre precio e indicador.
    
    Divergencia alcista: Precio hace LL, indicador hace HL
    Divergencia bajista: Precio hace HH, indicador hace LH
    """
    if indicator == 'rsi':
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
        rs = gain / (loss + 1e-10)
        ind = 100 - (100 / (1 + rs))
    elif indicator in df.columns:
        ind = df[indicator]
    else:
        raise ValueError(f"Indicador {indicator} no encontrado")
    
    # Swing points del precio
    price_swing_high = df['high'].rolling(lookback).max() == df['high']
    price_swing_low = df['low'].rolling(lookback).min() == df['low']
    
    # Swing points del indicador
    ind_swing_high = ind.rolling(lookback).max() == ind
    ind_swing_low = ind.rolling(lookback).min() == ind
    
    # Valores en swing points
    price_at_high = df['high'].where(price_swing_high)
    price_at_low = df['low'].where(price_swing_low)
    ind_at_high = ind.where(price_swing_high)
    ind_at_low = ind.where(price_swing_low)
    
    # Comparar con swing anterior
    prev_price_high = price_at_high.ffill().shift(lookback)
    prev_price_low = price_at_low.ffill().shift(lookback)
    prev_ind_high = ind_at_high.ffill().shift(lookback)
    prev_ind_low = ind_at_low.ffill().shift(lookback)
    
    # Divergencia bajista: precio HH pero indicador LH
    bearish_div = (
        price_swing_high & 
        (df['high'] > prev_price_high) & 
        (ind < prev_ind_high)
    )
    
    # Divergencia alcista: precio LL pero indicador HL
    bullish_div = (
        price_swing_low & 
        (df['low'] < prev_price_low) & 
        (ind > prev_ind_low)
    )
    
    return {
        f'div_{indicator}_bullish': bullish_div.astype(int),
        f'div_{indicator}_bearish': bearish_div.astype(int),
        f'div_{indicator}_signal': bullish_div.astype(int) - bearish_div.astype(int)
    }


# =============================================================================
# FUNCIÓN PRINCIPAL: CALCULAR TODOS LOS INDICADORES PRO
# =============================================================================

def calculate_pro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores profesionales.
    
    Args:
        df: DataFrame con OHLCV + timestamp
        
    Returns:
        DataFrame con todos los indicadores añadidos
    """
    logger.info("📊 Calculando indicadores profesionales...")
    
    df = df.copy()
    
    # 1. VWAP
    logger.info("   VWAP...")
    vwap_data = calc_vwap_bands(df)
    for k, v in vwap_data.items():
        df[k] = v
    
    # 2. CVD
    logger.info("   CVD...")
    cvd_data = calc_cvd_advanced(df)
    for k, v in cvd_data.items():
        df[k] = v
    
    # 3. OBV
    logger.info("   OBV...")
    obv_data = calc_obv_advanced(df)
    for k, v in obv_data.items():
        df[k] = v
    
    # 4. Fibonacci
    logger.info("   Fibonacci...")
    fib_data = calc_fibonacci_support_resistance(df)
    for k, v in fib_data.items():
        df[k] = v
    
    # 5. Market Structure
    logger.info("   Market Structure...")
    ms_data = calc_market_structure(df)
    for k, v in ms_data.items():
        df[k] = v
    
    # 6. Liquidity Zones
    logger.info("   Liquidity Zones...")
    liq_data = calc_liquidity_zones(df)
    for k, v in liq_data.items():
        df[k] = v
    
    # 7. Order Blocks
    logger.info("   Order Blocks...")
    ob_data = calc_order_blocks(df)
    for k, v in ob_data.items():
        df[k] = v
    
    # 8. Fair Value Gaps
    logger.info("   Fair Value Gaps...")
    fvg_data = calc_fair_value_gaps(df)
    for k, v in fvg_data.items():
        df[k] = v
    
    # 9. Divergencias
    logger.info("   Divergencias...")
    div_data = calc_divergences(df, 'rsi')
    for k, v in div_data.items():
        df[k] = v
    
    # Limpiar NaN
    df = df.ffill().bfill()
    
    # Contar features nuevas
    new_features = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
    logger.info(f"✅ {len(new_features)} indicadores profesionales calculados")
    
    return df


def get_pro_feature_columns() -> List[str]:
    """Retorna lista de columnas de features profesionales."""
    return [
        # VWAP
        'vwap_distance',
        
        # CVD
        'cvd_momentum', 'cvd_divergence', 'buy_volume_pct',
        
        # OBV
        'obv_momentum', 'obv_bullish_div', 'obv_bearish_div',
        
        # Fibonacci
        'fib_in_zone', 'fib_zone_type', 'fib_500_dist', 'fib_618_dist', 'fib_382_dist',
        
        # Market Structure
        'ms_hh', 'ms_hl', 'ms_lh', 'ms_ll', 'ms_trend',
        'ms_bullish_bos', 'ms_bearish_bos', 'ms_choch',
        
        # Liquidity
        'liq_vol_zscore', 'liq_high_vol_zone', 
        'liq_dist_high', 'liq_dist_low',
        
        # Order Blocks
        'ob_dist_bullish', 'ob_dist_bearish',
        'ob_near_bullish', 'ob_near_bearish',
        
        # FVG
        'fvg_bullish', 'fvg_bearish',
        'fvg_bullish_unfilled', 'fvg_bearish_unfilled',
        
        # Divergencias
        'div_rsi_bullish', 'div_rsi_bearish', 'div_rsi_signal'
    ]


# =============================================================================
# TEST
# =============================================================================

def test_pro_indicators():
    """Test de indicadores profesionales."""
    print("🧪 Test Indicadores Profesionales...")
    
    # Crear datos dummy
    np.random.seed(42)
    n = 1000
    
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='4H'),
        'open': prices + np.random.randn(n) * 50,
        'high': prices + np.abs(np.random.randn(n)) * 100,
        'low': prices - np.abs(np.random.randn(n)) * 100,
        'close': prices,
        'volume': np.random.exponential(1000, n)
    })
    
    # Calcular indicadores
    df_pro = calculate_pro_indicators(df)
    
    # Verificar
    pro_cols = get_pro_feature_columns()
    available = [c for c in pro_cols if c in df_pro.columns]
    
    print(f"✅ Features solicitadas: {len(pro_cols)}")
    print(f"✅ Features disponibles: {len(available)}")
    
    # Mostrar algunas estadísticas
    print(f"\n📊 Estadísticas de muestra:")
    for col in available[:5]:
        val = df_pro[col].dropna()
        print(f"   {col}: mean={val.mean():.4f}, std={val.std():.4f}")
    
    print("\n✅ Test completado")


if __name__ == "__main__":
    test_pro_indicators()