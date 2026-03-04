"""
Logger profesional para el bot de trading.
Registra todo: decisiones, señales, errores, estado del gate.
"""

import os
import sys
from datetime import datetime
from loguru import logger
from pathlib import Path


def setup_logger(log_dir: str = "logs", name: str = "bot"):
    """
    Configura el logger con rotación de archivos y formato profesional.
    
    Args:
        log_dir: Directorio donde guardar los logs
        name: Nombre base para los archivos de log
    """
    # Crear directorio si no existe
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Limpiar handlers existentes
    logger.remove()
    
    # Formato para consola (colorido)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Formato para archivo (más detallado)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Handler para consola
    logger.add(
        sys.stdout,
        format=console_format,
        level="INFO",
        colorize=True
    )
    
    # Handler para archivo general (rotación diaria)
    logger.add(
        f"{log_dir}/{name}_{{time:YYYY-MM-DD}}.log",
        format=file_format,
        level="DEBUG",
        rotation="00:00",  # Rotar a medianoche
        retention="30 days",
        compression="zip"
    )
    
    # Handler para errores (archivo separado)
    logger.add(
        f"{log_dir}/{name}_errors.log",
        format=file_format,
        level="ERROR",
        rotation="10 MB",
        retention="90 days"
    )
    
    # Handler para trades (archivo separado para auditoría)
    logger.add(
        f"{log_dir}/{name}_trades.log",
        format=file_format,
        level="INFO",
        filter=lambda record: "TRADE" in record["message"],
        rotation="1 week",
        retention="1 year"
    )
    
    logger.info(f"Logger inicializado. Logs en: {log_dir}/")
    
    return logger


def log_trade_decision(
    timestamp: datetime,
    close_price: float,
    action: int,
    pos_prev: int,
    pos_new: int,
    gate_on: bool,
    gate_dir: int,
    q10: float,
    q50: float,
    q90: float,
    reason: str = ""
):
    """
    Registra una decisión de trading con todos los detalles.
    Esto evita el problema de "no saber por qué el bot no opera".
    """
    action_map = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    
    msg = (
        f"TRADE_DECISION | "
        f"time={timestamp} | "
        f"close={close_price:.2f} | "
        f"action={action_map.get(action, action)} | "
        f"pos: {pos_prev}->{pos_new} | "
        f"gate_on={gate_on} | "
        f"gate_dir={gate_dir} | "
        f"q10={q10:.6f} q50={q50:.6f} q90={q90:.6f}"
    )
    
    if reason:
        msg += f" | reason={reason}"
    
    if pos_prev != pos_new:
        logger.info(f"🔄 TRADE EXECUTED | {msg}")
    else:
        logger.debug(f"📊 {msg}")


def log_gate_status(
    vol: float,
    vol_min: float,
    thr: float,
    thr_base: float,
    gate_on: bool,
    gate_dir: int,
    block_reason: str = ""
):
    """
    Registra el estado del gate para diagnóstico.
    Esto resuelve el problema de "no saber por qué gate_on=0".
    """
    status = "✅ OPEN" if gate_on else "🚫 BLOCKED"
    
    msg = (
        f"GATE_STATUS | {status} | "
        f"vol={vol:.6f} (min={vol_min:.6f}) | "
        f"thr={thr:.6f} (base={thr_base:.6f}) | "
        f"dir={gate_dir}"
    )
    
    if block_reason:
        msg += f" | block_reason={block_reason}"
    
    if gate_on:
        logger.info(msg)
    else:
        logger.warning(msg)


def log_order_execution(
    side: str,
    symbol: str,
    quantity: float,
    price: float,
    order_id: str,
    status: str
):
    """
    Registra la ejecución de órdenes en Binance.
    """
    logger.info(
        f"🎯 ORDER | {side} {quantity:.8f} {symbol} @ {price:.2f} | "
        f"id={order_id} | status={status}"
    )


def log_error_with_context(error: Exception, context: dict):
    """
    Registra errores con contexto completo para debugging.
    """
    logger.error(f"❌ ERROR: {type(error).__name__}: {error}")
    logger.error(f"   Context: {context}")


# Crear instancia global del logger
bot_logger = setup_logger()
