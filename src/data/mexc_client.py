"""
Cliente MEXC para trading de Futuros.
Soporta Long y Short con apalancamiento.
"""

import os
import time
import hmac
import hashlib
from datetime import datetime
from typing import Optional, Dict, Tuple
import requests
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class MEXCFuturesClient:
    """
    Cliente para MEXC Futuros.
    Documentación: https://mexcdevelop.github.io/apidocs/contract_v1_en/
    """
    
    BASE_URL = "https://contract.mexc.com"
    
    def __init__(self):
        """Inicializa el cliente MEXC Futuros."""
        self.api_key = os.getenv("MEXC_API_KEY", "")
        self.api_secret = os.getenv("MEXC_API_SECRET", "")
        
        if not self.api_key or not self.api_secret:
            logger.warning("⚠️ MEXC API keys no configuradas. Solo modo lectura.")
        else:
            logger.info("✅ Cliente MEXC Futuros inicializado")
        
        self._verify_connection()
    
    def _verify_connection(self):
        """Verifica conexión a MEXC."""
        try:
            response = self._request("GET", "/api/v1/contract/ping")
            logger.info("✅ Conexión MEXC OK")
        except Exception as e:
            logger.error(f"❌ Error conectando a MEXC: {e}")
    
    def _generate_signature(self, params: dict) -> str:
        """Genera firma HMAC SHA256 para requests autenticados."""
        # Ordenar parámetros y crear query string
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
        
        # Crear firma
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: dict = None,
        signed: bool = False
    ) -> dict:
        """
        Realiza una petición a la API de MEXC.
        
        Args:
            method: GET, POST, DELETE
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            signed: Si requiere autenticación
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        headers = {
            "Content-Type": "application/json"
        }
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
            headers['ApiKey'] = self.api_key
            headers['Request-Time'] = str(params['timestamp'])
            headers['Signature'] = params['signature']
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=10)
            elif method == "DELETE":
                response = requests.delete(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Método no soportado: {method}")
            
            response.raise_for_status()
            data = response.json()
            
            # MEXC retorna success=true/false
            if isinstance(data, dict) and data.get('success') == False:
                raise Exception(f"MEXC Error: {data.get('message', data)}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request error: {e}")
            raise
    
    # ========== DATOS DE MERCADO ==========
    
    def get_ticker(self, symbol: str = "BTC_USDT") -> dict:
        """Obtiene precio actual del par."""
        response = self._request("GET", f"/api/v1/contract/ticker?symbol={symbol}")
        
        if response.get('success') and response.get('data'):
            data = response['data']
            logger.debug(f"💵 MEXC {symbol}: {data.get('lastPrice')}")
            return data
        
        return {}
    
    def get_current_price(self, symbol: str = "BTC_USDT") -> float:
        """Obtiene solo el precio actual."""
        ticker = self.get_ticker(symbol)
        price = float(ticker.get('lastPrice', 0))
        return price
    
    def get_klines(
        self,
        symbol: str = "BTC_USDT",
        interval: str = "Hour1",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Obtiene velas históricas de MEXC Futuros.
        Si falla (403), retorna DataFrame vacío (usaremos Binance para datos).
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params['start'] = start_time
        if end_time:
            params['end'] = end_time
        
        try:
            response = self._request("GET", "/api/v1/contract/kline", params)
            
            if not response.get('success') or not response.get('data'):
                logger.warning(f"⚠️ No se obtuvieron datos de MEXC klines")
                return pd.DataFrame()
            
            data = response['data']
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.debug(f"📊 MEXC: Obtenidas {len(df)} velas {interval}")
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ MEXC klines no disponible (usaremos Binance para datos): {e}")
            return pd.DataFrame()
    
    # ========== CUENTA ==========
    
    def get_account_info(self) -> dict:
        """Obtiene información de la cuenta de futuros."""
        response = self._request("GET", "/api/v1/private/account/assets", signed=True)
        
        if response.get('success') and response.get('data'):
            return response['data']
        
        return {}
    
    def get_balance(self, currency: str = "USDT") -> Tuple[float, float]:
        """
        Obtiene balance disponible y en uso.
        
        Returns:
            Tuple (available, frozen)
        """
        account = self.get_account_info()
        
        for asset in account:
            if asset.get('currency') == currency:
                available = float(asset.get('availableBalance', 0))
                frozen = float(asset.get('frozenBalance', 0))
                logger.debug(f"💰 MEXC Balance {currency}: available={available:.2f}, frozen={frozen:.2f}")
                return available, frozen
        
        return 0.0, 0.0
    
    # ========== POSICIONES ==========
    
    def get_positions(self, symbol: str = "BTC_USDT") -> list:
        """Obtiene posiciones abiertas."""
        response = self._request(
            "GET", 
            f"/api/v1/private/position/open_positions?symbol={symbol}",
            signed=True
        )
        
        if response.get('success') and response.get('data'):
            positions = response['data']
            logger.debug(f"📊 Posiciones abiertas: {len(positions)}")
            return positions
        
        return []
    
    def get_current_position(self, symbol: str = "BTC_USDT") -> Optional[dict]:
        """
        Obtiene la posición actual del símbolo.
        
        Returns:
            Dict con posición o None si no hay
        """
        positions = self.get_positions(symbol)
        
        for pos in positions:
            if pos.get('symbol') == symbol and float(pos.get('holdVol', 0)) > 0:
                return {
                    'symbol': pos['symbol'],
                    'side': pos.get('positionType'),  # 1=Long, 2=Short
                    'size': float(pos.get('holdVol', 0)),
                    'entry_price': float(pos.get('openAvgPrice', 0)),
                    'mark_price': float(pos.get('markPrice', 0)),
                    'pnl': float(pos.get('unrealised', 0)),
                    'leverage': int(pos.get('leverage', 1))
                }
        
        return None
    
    # ========== ÓRDENES ==========
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Configura el apalancamiento para un símbolo."""
        params = {
            "symbol": symbol,
            "leverage": leverage,
            "openType": 1  # 1=isolated, 2=cross
        }
        
        try:
            response = self._request("POST", "/api/v1/private/position/change_leverage", params, signed=True)
            logger.info(f"⚙️ Leverage configurado: {symbol} = {leverage}x")
            return True
        except Exception as e:
            logger.error(f"❌ Error configurando leverage: {e}")
            return False
    
    def place_market_order(
        self,
        symbol: str,
        side: int,      # 1=Open Long, 2=Close Short, 3=Open Short, 4=Close Long
        vol: float,     # Cantidad en contratos
        leverage: int = 3
    ) -> Optional[dict]:
        """
        Coloca una orden de mercado.
        
        Args:
            symbol: Par de trading (BTC_USDT)
            side: 1=Open Long, 2=Close Short, 3=Open Short, 4=Close Long
            vol: Cantidad
            leverage: Apalancamiento
            
        Returns:
            Respuesta de la orden o None si falla
        """
        side_names = {1: "OPEN_LONG", 2: "CLOSE_SHORT", 3: "OPEN_SHORT", 4: "CLOSE_LONG"}
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": 5,  # 5 = Market order
            "vol": vol,
            "leverage": leverage,
            "openType": 1  # 1=isolated
        }
        
        try:
            logger.info(f"📤 Enviando orden: {side_names.get(side)} {vol} {symbol} @ market")
            
            response = self._request("POST", "/api/v1/private/order/submit", params, signed=True)
            
            if response.get('success') and response.get('data'):
                order_id = response['data'].get('orderId')
                logger.info(f"✅ Orden ejecutada: {order_id}")
                return response['data']
            else:
                logger.error(f"❌ Orden fallida: {response}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en orden: {e}")
            return None
    
    def open_long(self, symbol: str, vol: float, leverage: int = 3) -> Optional[dict]:
        """Abre posición LONG."""
        return self.place_market_order(symbol, side=1, vol=vol, leverage=leverage)
    
    def close_long(self, symbol: str, vol: float) -> Optional[dict]:
        """Cierra posición LONG."""
        return self.place_market_order(symbol, side=4, vol=vol, leverage=1)
    
    def open_short(self, symbol: str, vol: float, leverage: int = 3) -> Optional[dict]:
        """Abre posición SHORT."""
        return self.place_market_order(symbol, side=3, vol=vol, leverage=leverage)
    
    def close_short(self, symbol: str, vol: float) -> Optional[dict]:
        """Cierra posición SHORT."""
        return self.place_market_order(symbol, side=2, vol=vol, leverage=1)
    
    def close_position(self, symbol: str = "BTC_USDT") -> bool:
        """Cierra cualquier posición abierta del símbolo."""
        position = self.get_current_position(symbol)
        
        if not position:
            logger.info("📭 No hay posición abierta para cerrar")
            return True
        
        size = position['size']
        side = position['side']  # 1=Long, 2=Short
        
        if side == 1:  # Long
            result = self.close_long(symbol, size)
        else:  # Short
            result = self.close_short(symbol, size)
        
        return result is not None
    
    # ========== STOP LOSS / TAKE PROFIT ==========
    
    def set_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        side: int  # 1=para long, 2=para short
    ) -> Optional[dict]:
        """Configura stop loss para la posición."""
        params = {
            "symbol": symbol,
            "stopLossPrice": stop_price,
            "positionType": side
        }
        
        try:
            response = self._request("POST", "/api/v1/private/position/change_stop_loss", params, signed=True)
            logger.info(f"🛑 Stop Loss configurado: {symbol} @ {stop_price}")
            return response.get('data')
        except Exception as e:
            logger.error(f"❌ Error configurando SL: {e}")
            return None
    
    def set_take_profit(
        self,
        symbol: str,
        take_price: float,
        side: int  # 1=para long, 2=para short
    ) -> Optional[dict]:
        """Configura take profit para la posición."""
        params = {
            "symbol": symbol,
            "takeProfitPrice": take_price,
            "positionType": side
        }
        
        try:
            response = self._request("POST", "/api/v1/private/position/change_take_profit", params, signed=True)
            logger.info(f"🎯 Take Profit configurado: {symbol} @ {take_price}")
            return response.get('data')
        except Exception as e:
            logger.error(f"❌ Error configurando TP: {e}")
            return None


def test_mexc_connection():
    """Prueba la conexión a MEXC."""
    print("=" * 50)
    print("TEST CONEXIÓN MEXC FUTUROS")
    print("=" * 50)
    
    client = MEXCFuturesClient()
    
    # Test 1: Precio actual
    print("\n1. Precio actual BTC_USDT:")
    price = client.get_current_price("BTC_USDT")
    print(f"   ${price:,.2f}")
    
    # Test 2: Ticker completo
    print("\n2. Ticker completo:")
    ticker = client.get_ticker("BTC_USDT")
    if ticker:
        print(f"   Last: {ticker.get('lastPrice')}")
        print(f"   24h Vol: {ticker.get('volume24')}")
    
    # Test 3: Klines (puede fallar por región)
    print("\n3. Klines (puede no estar disponible por región):")
    df = client.get_klines("BTC_USDT", "Hour1", limit=5)
    if not df.empty:
        print(df.to_string())
    else:
        print("   ⚠️ Klines no disponible - usaremos Binance para datos históricos")
    
    # Test 4: Balance (solo si hay API keys)
    if client.api_key:
        print("\n4. Balance cuenta:")
        try:
            available, frozen = client.get_balance("USDT")
            print(f"   Disponible: ${available:.2f}")
            print(f"   Bloqueado: ${frozen:.2f}")
        except Exception as e:
            print(f"   ⚠️ No se pudo obtener balance: {e}")
        
        print("\n5. Posiciones abiertas:")
        try:
            position = client.get_current_position("BTC_USDT")
            if position:
                print(f"   {position}")
            else:
                print("   No hay posiciones abiertas")
        except Exception as e:
            print(f"   ⚠️ No se pudo obtener posiciones: {e}")
    
    print("\n" + "=" * 50)
    print("✅ TEST COMPLETADO")
    print("=" * 50)
    print("\n📝 NOTA: Los datos históricos se obtendrán de Binance.")
    print("   MEXC se usará solo para ejecutar trades.")


if __name__ == "__main__":
    test_mexc_connection()



