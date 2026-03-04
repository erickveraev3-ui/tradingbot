"""
Cliente de Binance para obtener datos en tiempo real.
NUNCA depende de CSV estáticos - siempre consulta la API.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from loguru import logger

# Cargar variables de entorno
load_dotenv()


class BinanceDataClient:
    """
    Cliente para obtener datos de Binance.
    Diseñado para evitar el error de "bot ciego por CSV congelado".
    """
    
    def __init__(self, testnet: bool = False):
        """
        Inicializa el cliente de Binance.
        
        Args:
            testnet: Si True, usa el testnet de Binance
        """
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            logger.warning("⚠️ API keys no encontradas. Solo modo lectura disponible.")
            self.client = Client("", "")
        else:
            self.client = Client(self.api_key, self.api_secret, testnet=testnet)
            logger.info(f"✅ Cliente Binance inicializado (testnet={testnet})")
        
        # Verificar conexión
        self._verify_connection()
    
    def _verify_connection(self):
        """Verifica que la conexión a Binance funcione."""
        try:
            server_time = self.client.get_server_time()
            server_dt = datetime.fromtimestamp(server_time['serverTime'] / 1000)
            logger.info(f"✅ Conexión Binance OK. Server time: {server_dt}")
        except BinanceAPIException as e:
            logger.error(f"❌ Error conectando a Binance: {e}")
            raise
    
    def get_klines(
        self,
        symbol: str = "BTCUSDC",
        interval: str = "1h",
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Obtiene velas históricas de Binance.
        
        Args:
            symbol: Par de trading (ej: "BTCUSDC")
            interval: Intervalo de velas ("1h", "15m", etc.)
            limit: Número máximo de velas (max 1000)
            start_time: Fecha inicio (opcional)
            end_time: Fecha fin (opcional)
            
        Returns:
            DataFrame con columnas: timestamp, open, high, low, close, volume
        """
        try:
            # Preparar parámetros
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
            
            # Obtener datos
            klines = self.client.get_klines(**params)
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Procesar columnas
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Convertir a float
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)
            
            # Seleccionar columnas relevantes
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
            
            logger.debug(f"📊 Obtenidas {len(df)} velas {interval} de {symbol}")
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"❌ Error obteniendo klines: {e}")
            raise
    
    def get_historical_klines(
        self,
        symbol: str = "BTCUSDC",
        interval: str = "1h",
        start_date: str = "1 Jan 2023",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Obtiene datos históricos extensos (más de 1000 velas).
        Hace múltiples llamadas a la API si es necesario.
        
        Args:
            symbol: Par de trading
            interval: Intervalo de velas
            start_date: Fecha inicio en formato legible
            end_date: Fecha fin (None = ahora)
            
        Returns:
            DataFrame con todo el histórico
        """
        try:
            logger.info(f"📥 Descargando histórico {symbol} {interval} desde {start_date}...")
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
            
            logger.info(f"✅ Descargadas {len(df)} velas. Rango: {df['timestamp'].min()} a {df['timestamp'].max()}")
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"❌ Error descargando histórico: {e}")
            raise
    
    def get_latest_closed_candle(
        self,
        symbol: str = "BTCUSDC",
        interval: str = "1h"
    ) -> pd.Series:
        """
        Obtiene la última vela CERRADA (no la actual en formación).
        Esto es crucial para evitar señales con datos incompletos.
        
        Returns:
            Serie con los datos de la última vela cerrada
        """
        # Obtener las últimas 2 velas
        df = self.get_klines(symbol=symbol, interval=interval, limit=2)
        
        # La última fila es la vela actual (en formación)
        # La penúltima es la última cerrada
        if len(df) >= 2:
            candle = df.iloc[-2]
            logger.debug(f"🕯️ Última vela cerrada: {candle['timestamp']} close={candle['close']:.2f}")
            return candle
        else:
            raise ValueError("No se pudieron obtener suficientes velas")
    
    def get_account_balance(self, asset: str = "USDC") -> Tuple[float, float]:
        """
        Obtiene el balance de un activo.
        
        Returns:
            Tuple (free, locked)
        """
        try:
            account = self.client.get_account()
            
            for balance in account['balances']:
                if balance['asset'] == asset:
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    logger.debug(f"💰 Balance {asset}: free={free:.4f}, locked={locked:.4f}")
                    return free, locked
            
            return 0.0, 0.0
            
        except BinanceAPIException as e:
            logger.error(f"❌ Error obteniendo balance: {e}")
            raise
    
    def get_current_price(self, symbol: str = "BTCUSDC") -> float:
        """Obtiene el precio actual del par."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            logger.debug(f"💵 Precio actual {symbol}: {price:.2f}")
            return price
        except BinanceAPIException as e:
            logger.error(f"❌ Error obteniendo precio: {e}")
            raise
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" o "SELL"
        quantity: float
    ) -> dict:
        """
        Coloca una orden de mercado.
        
        Args:
            symbol: Par de trading
            side: "BUY" o "SELL"
            quantity: Cantidad a operar
            
        Returns:
            Respuesta de la orden
        """
        try:
            logger.info(f"📤 Enviando orden: {side} {quantity:.8f} {symbol}")
            
            order = self.client.order_market(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            
            logger.info(f"✅ Orden ejecutada: {order['orderId']} status={order['status']}")
            
            return order
            
        except BinanceAPIException as e:
            logger.error(f"❌ Error en orden: {e}")
            raise
    
    def get_symbol_info(self, symbol: str = "BTCUSDC") -> dict:
        """
        Obtiene información del símbolo (precisión, límites, etc.).
        Importante para formatear correctamente las órdenes.
        """
        try:
            info = self.client.get_symbol_info(symbol)
            
            # Extraer filtros importantes
            filters = {f['filterType']: f for f in info['filters']}
            
            result = {
                'symbol': symbol,
                'base_asset': info['baseAsset'],
                'quote_asset': info['quoteAsset'],
                'base_precision': info['baseAssetPrecision'],
                'quote_precision': info['quoteAssetPrecision'],
                'min_qty': float(filters['LOT_SIZE']['minQty']),
                'max_qty': float(filters['LOT_SIZE']['maxQty']),
                'step_size': float(filters['LOT_SIZE']['stepSize']),
                'min_notional': float(filters.get('NOTIONAL', {}).get('minNotional', 0))
            }
            
            logger.debug(f"ℹ️ Info {symbol}: min_qty={result['min_qty']}, step={result['step_size']}")
            
            return result
            
        except BinanceAPIException as e:
            logger.error(f"❌ Error obteniendo info del símbolo: {e}")
            raise


# Función de utilidad para pruebas
def test_connection():
    """Prueba rápida de conexión."""
    client = BinanceDataClient()
    
    # Test 1: Precio actual
    price = client.get_current_price("BTCUSDC")
    print(f"Precio BTC/USDC: ${price:,.2f}")
    
    # Test 2: Última vela cerrada
    candle = client.get_latest_closed_candle("BTCUSDC", "1h")
    print(f"Última vela 1H cerrada: {candle['timestamp']}")
    print(f"  O={candle['open']:.2f} H={candle['high']:.2f} L={candle['low']:.2f} C={candle['close']:.2f}")
    
    # Test 3: Info del símbolo
    info = client.get_symbol_info("BTCUSDC")
    print(f"Min order: {info['min_qty']} BTC")


if __name__ == "__main__":
    test_connection()
