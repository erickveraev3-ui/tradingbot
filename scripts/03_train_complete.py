"""
Script 03: Entrenamiento Completo del Sistema Multi-Timeframe.

Entrena:
1. MultiTimeframeDetector (3 LSTMs + Regime Detection)
2. SACAdvanced (con Kelly + Trailing Stop)

Uso:
    python scripts/03_train_complete.py
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from loguru import logger

from src.features.indicators import calculate_all_indicators
from src.models.multi_timeframe_detector import (
    MultiTimeframeDetector,
    MultiTimeframeDataset,
    MultiTimeframeTrainer
)
from src.models.sac_advanced import (
    SACAdvanced,
    AdvancedTradingEnv,
    AdvancedTradingConfig
)


def load_and_prepare_data():
    """Carga y prepara datos de las 3 temporalidades."""
    logger.info("📥 Cargando datos...")
    
    # Cargar CSVs
    df_1h = pd.read_csv(root_dir / "data/raw/btcusdt_1h.csv")
    df_4h = pd.read_csv(root_dir / "data/raw/btcusdt_4h.csv")
    df_1d = pd.read_csv(root_dir / "data/raw/btcusdt_1d.csv")
    
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
    
    logger.info(f"   1H: {len(df_1h):,} velas")
    logger.info(f"   4H: {len(df_4h):,} velas")
    logger.info(f"   1D: {len(df_1d):,} velas")
    
    # Calcular indicadores para cada timeframe
    logger.info("📊 Calculando indicadores...")
    df_1h = calculate_all_indicators(df_1h)
    df_4h = calculate_all_indicators(df_4h)
    df_1d = calculate_all_indicators(df_1d)
    
    # Calcular ATR para trailing stop (en 4H que es nuestro timeframe principal)
    if 'atr' not in df_4h.columns:
        high = df_4h['high'].values
        low = df_4h['low'].values
        close = df_4h['close'].values
        
        tr = np.maximum(high[1:] - low[1:], 
                       np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
        tr = np.insert(tr, 0, high[0] - low[0])
        
        atr = pd.Series(tr).rolling(14).mean().values
        df_4h['atr'] = atr
    
    return df_1h, df_4h, df_1d


def train_detector(df_1h, df_4h, df_1d, device):
    """Entrena el detector multi-timeframe."""
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 1: ENTRENAMIENTO MULTI-TIMEFRAME DETECTOR")
    logger.info("=" * 60)
    
    # Crear dataset
    logger.info("\n📦 Creando dataset multi-timeframe...")
    dataset = MultiTimeframeDataset(df_1h, df_4h, df_1d)
    
    # Split 80/20
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    logger.info(f"   Train: {len(train_dataset):,}")
    logger.info(f"   Test: {len(test_dataset):,}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Crear modelo
    sample = dataset[0]
    input_dim_1d = sample['x_1d'].shape[1]
    input_dim_4h = sample['x_4h'].shape[1]
    input_dim_1h = sample['x_1h'].shape[1]
    
    model = MultiTimeframeDetector(
        input_dim_1d=input_dim_1d,
        input_dim_4h=input_dim_4h,
        input_dim_1h=input_dim_1h,
        hidden_dim=128,
        embedding_dim_1d=64,
        embedding_dim_4h=64,
        embedding_dim_1h=32,
        dropout=0.2
    )
    
    # Entrenar
    trainer = MultiTimeframeTrainer(model, lr=1e-3, device=device)
    history = trainer.train(train_loader, test_loader, epochs=100, early_stopping=15)
    
    # Guardar
    save_path = root_dir / "artifacts/multi_timeframe/detector.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(save_path))
    
    # Evaluar
    val_metrics = trainer.validate(test_loader)
    logger.info(f"\n📊 Métricas finales detector:")
    logger.info(f"   Val Loss: {val_metrics['val_loss']:.4f}")
    logger.info(f"   Regime Accuracy: {val_metrics['regime_accuracy']:.1%}")
    
    return model, dataset


def generate_embeddings_and_predictions(model, dataset, device):
    """Genera embeddings y predicciones para todo el dataset."""
    logger.info("\n🧠 Generando embeddings y predicciones...")
    
    model.eval()
    model.to(device)
    
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    all_embeddings = []
    all_pred_4h = []
    all_pred_12h = []
    all_pred_24h = []
    all_regimes = []
    all_prices = []
    
    with torch.no_grad():
        for batch in loader:
            x_1d = batch['x_1d'].to(device)
            x_4h = batch['x_4h'].to(device)
            x_1h = batch['x_1h'].to(device)
            
            outputs = model(x_1d, x_4h, x_1h)
            
            all_embeddings.append(outputs['embedding'].cpu().numpy())
            all_pred_4h.append(outputs['pred_4h'].cpu().numpy())
            all_pred_12h.append(outputs['pred_12h'].cpu().numpy())
            all_pred_24h.append(outputs['pred_24h'].cpu().numpy())
            all_regimes.append(outputs['regime_logits'].argmax(dim=-1).cpu().numpy())
            all_prices.append(batch['price'].numpy())
    
    embeddings = np.vstack(all_embeddings)
    pred_4h = np.vstack(all_pred_4h)
    pred_12h = np.vstack(all_pred_12h)
    pred_24h = np.vstack(all_pred_24h)
    regimes = np.concatenate(all_regimes)
    prices = np.concatenate(all_prices)
    
    logger.info(f"   Embeddings: {embeddings.shape}")
    logger.info(f"   Predictions: {pred_12h.shape}")
    logger.info(f"   Regimes distribution: {np.bincount(regimes)}")
    
    return embeddings, pred_4h, pred_12h, pred_24h, regimes, prices


def train_sac(embeddings, pred_4h, pred_12h, pred_24h, regimes, prices, atrs, device):
    """Entrena el agente SAC avanzado."""
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 2: ENTRENAMIENTO SAC ADVANCED")
    logger.info("=" * 60)
    
    # Split 80/20
    split_idx = int(len(embeddings) * 0.8)
    
    train_data = {
        'embeddings': embeddings[:split_idx],
        'pred_4h': pred_4h[:split_idx],
        'pred_12h': pred_12h[:split_idx],
        'pred_24h': pred_24h[:split_idx],
        'regimes': regimes[:split_idx],
        'prices': prices[:split_idx],
        'atrs': atrs[:split_idx]
    }
    
    test_data = {
        'embeddings': embeddings[split_idx:],
        'pred_4h': pred_4h[split_idx:],
        'pred_12h': pred_12h[split_idx:],
        'pred_24h': pred_24h[split_idx:],
        'regimes': regimes[split_idx:],
        'prices': prices[split_idx:],
        'atrs': atrs[split_idx:]
    }
    
    logger.info(f"   Train: {len(train_data['embeddings']):,}")
    logger.info(f"   Test: {len(test_data['embeddings']):,}")
    
    # Configuración
    config = AdvancedTradingConfig(
        initial_capital=10000.0,
        max_leverage=3,
        kelly_fraction=0.5,
        atr_multiplier_stop=2.0,
        atr_multiplier_trail=1.5,
        max_drawdown=0.15
    )
    
    # Crear entornos
    train_env = AdvancedTradingEnv(
        train_data['embeddings'],
        train_data['pred_4h'],
        train_data['pred_12h'],
        train_data['pred_24h'],
        train_data['regimes'],
        train_data['prices'],
        train_data['atrs'],
        config
    )
    
    test_env = AdvancedTradingEnv(
        test_data['embeddings'],
        test_data['pred_4h'],
        test_data['pred_12h'],
        test_data['pred_24h'],
        test_data['regimes'],
        test_data['prices'],
        test_data['atrs'],
        config
    )
    
    # Crear agente
    agent = SACAdvanced(
        state_dim=train_env.state_dim,
        hidden_dim=256,
        n_regimes=4,
        lr=3e-4,
        device=device
    )
    
    # Entrenar
    history = agent.train(
        env=train_env,
        total_steps=500000,
        batch_size=256,
        start_steps=10000,
        log_interval=25000,
        eval_interval=50000
    )
    
    # Evaluar en test
    logger.info("\n📊 Evaluación en TEST SET...")
    test_metrics = agent.evaluate(test_env, n_episodes=20)
    
    # Guardar
    save_path = root_dir / "artifacts/sac_advanced/agent.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_path))
    
    return agent, test_metrics, history


def main():
    print("\n" + "=" * 60)
    print("   🚀 ENTRENAMIENTO COMPLETO MULTI-TIMEFRAME + SAC")
    print("=" * 60 + "\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Device: {device}")
    if device == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar datos
    df_1h, df_4h, df_1d = load_and_prepare_data()
    
    # Fase 1: Entrenar detector
    detector, dataset = train_detector(df_1h, df_4h, df_1d, device)
    
    # Generar embeddings
    embeddings, pred_4h, pred_12h, pred_24h, regimes, prices = \
        generate_embeddings_and_predictions(detector, dataset, device)
    
    # ATRs
    atrs = df_4h['atr'].values[-len(prices):]
    atrs = np.nan_to_num(atrs, nan=np.nanmean(atrs))
    
    # Fase 2: Entrenar SAC
    agent, test_metrics, history = train_sac(
        embeddings, pred_4h, pred_12h, pred_24h, regimes, prices, atrs, device
    )
    
    # Resumen final
    print("\n" + "=" * 60)
    print("   ✅ ENTRENAMIENTO COMPLETO FINALIZADO")
    print("=" * 60)
    
    print(f"\n📊 RESULTADOS EN TEST SET (20 episodios):")
    print(f"   Retorno medio: {test_metrics['mean_return']*100:+.2f}%")
    print(f"   Retorno std: {test_metrics['std_return']*100:.2f}%")
    print(f"   Sharpe ratio: {test_metrics['sharpe']:.2f}")
    print(f"   Win rate: {test_metrics['mean_win_rate']*100:.1f}%")
    print(f"   Max drawdown: {test_metrics['max_drawdown']*100:.1f}%")
    print(f"   Trades promedio: {test_metrics['mean_trades']:.0f}")
    
    print(f"\n💾 Modelos guardados:")
    print(f"   Detector: artifacts/multi_timeframe/detector.pt")
    print(f"   SAC Agent: artifacts/sac_advanced/agent.pt")
    
    # Guardar métricas
    metrics_path = root_dir / "artifacts/training_metrics.npz"
    np.savez(
        metrics_path,
        test_return=test_metrics['mean_return'],
        test_sharpe=test_metrics['sharpe'],
        test_winrate=test_metrics['mean_win_rate'],
        test_maxdd=test_metrics['max_drawdown'],
        history_returns=history.get('returns', []),
        history_win_rates=history.get('win_rates', [])
    )
    
    print(f"\n🎉 Siguiente paso: python scripts/04_backtest.py")


if __name__ == "__main__":
    main()
