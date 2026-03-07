"""
Generador de Señales de Trading.
Combina múltiples indicadores para generar señales de entrada/salida.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from loguru import logger
from pathlib import Path

# Importar indicadores
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.indicators import calculate_all_indicators


class SignalGenerator:
    """
    Genera señales de trading basadas en múltiples indicadores.
    
    Estrategia:
    - Tendencia: EMA 21 > EMA 55 > EMA 200 (alcista) o inverso (bajista)
    - Entrada: RSI + MACD + ADX confirman
    - Filtro: ADX > 25 (tendencia fuerte)
    """
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config: Diccionario de configuración (opcional)
        """
        self.config = config or {}
        
        # Parámetros por defecto
        self.rsi_oversold = self.config.get('rsi_oversold', 35)
        self.rsi_overbought = self.config.get('rsi_overbought', 65)
        self.adx_threshold = self.config.get('adx_threshold', 20)
        self.min_score = self.config.get('min_score', 3)
        
        logger.info(f"📊 SignalGenerator inicializado")
        logger.info(f"   RSI: oversold={self.rsi_oversold}, overbought={self.rsi_overbought}")
        logger.info(f"   ADX threshold: {self.adx_threshold}")
        logger.info(f"   Min score para señal: {self.min_score}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de trading para cada fila.
        
        Args:
            df: DataFrame con indicadores calculados
            
        Returns:
            DataFrame con columnas de señales añadidas
        """
        df = df.copy()
        
        # ========== SEÑALES INDIVIDUALES ==========
        
        # 1. Tendencia EMA (peso: 2)
        # Alcista: EMA21 > EMA55 > EMA200
        # Bajista: EMA21 < EMA55 < EMA200
        df['sig_ema_trend'] = np.where(
            (df['ema_21'] > df['ema_55']) & (df['ema_55'] > df['ema_200']), 2,
            np.where(
                (df['ema_21'] < df['ema_55']) & (df['ema_55'] < df['ema_200']), -2,
                0
            )
        )
        
        # 2. Precio respecto a EMA 200 (peso: 1)
        df['sig_ema200'] = np.where(df['close'] > df['ema_200'], 1, -1)
        
        # 3. RSI (peso: 1)
        df['sig_rsi'] = np.where(
            df['rsi'] < self.rsi_oversold, 1,      # Sobrevendido = comprar
            np.where(df['rsi'] > self.rsi_overbought, -1, 0)  # Sobrecomprado = vender
        )
        
        # 4. MACD cruce (peso: 2)
        df['sig_macd'] = np.where(
            (df['macd'] > df['macd_signal']) & (df['macd_hist'] > 0), 2,
            np.where(
                (df['macd'] < df['macd_signal']) & (df['macd_hist'] < 0), -2,
                0
            )
        )
        
        # 5. MACD histograma creciente (peso: 1)
