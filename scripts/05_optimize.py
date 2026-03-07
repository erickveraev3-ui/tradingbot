"""
Script 05: Optimización de Hiperparámetros con Optuna.
Encuentra la configuración óptima para maximizar Sharpe Ratio.

Uso:
    python scripts/05_optimize.py
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from loguru import logger
from datetime import datetime

from src.features.indicators import calculate_all_indicators
from src.models.multi_timeframe_detector import MultiTimeframeDetector, MultiTimeframeDataset
from src.models.sac_advanced import SACAdvanced, AdvancedTradingEnv, AdvancedTradingConfig


# Variables globales para no recargar datos en cada trial
GLOBAL_DATA = None


def load_data_once():
    """Carga datos una sola vez."""
    global GLOBAL_DATA
    
    if GLOBAL_DATA is not None:
        return GLOBAL_DATA
    
    logger.info("📥 Cargando datos (una vez)...")
    
    df_1h = pd.read_csv(root_dir / "data/raw/btcusdt_1h.csv")
    df_4h = pd.read_csv(root_dir / "data/raw/btcusdt_4h.csv")
    df_1d = pd.read_csv(root_dir / "data/raw/btcusdt_1d.csv")
    
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
    
    df_1h = calculate_all_indicators(df_1h)
    df_4h = calculate_all_indicators(df_4h)
    df_1d = calculate_all_indicators(df_1d)
    
    # ATR
    if 'atr' not in df_4h.columns:
        high, low, close = df_4h['high'].values, df_4h['low'].values, df_4h['close'].values
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        tr = np.insert(tr, 0, high[0] - low[0])
        df_4h['atr'] = pd.Series(tr).rolling(14).mean().values
    
    # Cargar detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector_path = root_dir / "artifacts/multi_timeframe/detector.pt"
    detector_checkpoint = torch.load(detector_path, map_location=device)
    
    detector = MultiTimeframeDetector(input_dim_1d=14, input_dim_4h=14, input_dim_1h=14)
    detector.load_state_dict(detector_checkpoint['model_state'])
    detector.to(device)
    detector.eval()
    
    # Crear dataset
    dataset = MultiTimeframeDataset(df_1h, df_4h, df_1d)
    
    # Generar embeddings para TODO el dataset
    logger.info("🧠 Generando embeddings...")
    
    all_embeddings = []
    all_pred_4h = []
    all_pred_12h = []
    all_pred_24h = []
    all_regimes = []
    all_prices = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x_1d = sample['x_1d'].unsqueeze(0).to(device)
            x_4h = sample['x_4h'].unsqueeze(0).to(device)
            x_1h = sample['x_1h'].unsqueeze(0).to(device)
            
            outputs = detector(x_1d, x_4h, x_1h)
            
            all_embeddings.append(outputs['embedding'].cpu().numpy())
            all_pred_4h.append(outputs['pred_4h'].cpu().numpy())
            all_pred_12h.append(outputs['pred_12h'].cpu().numpy())
            all_pred_24h.append(outputs['pred_24h'].cpu().numpy())
            all_regimes.append(outputs['regime_logits'].argmax(dim=-1).cpu().numpy())
            all_prices.append(sample['price'].numpy())
    
    embeddings = np.vstack(all_embeddings)
    pred_4h = np.vstack(all_pred_4h)
    pred_12h = np.vstack(all_pred_12h)
    pred_24h = np.vstack(all_pred_24h)
    regimes = np.concatenate(all_regimes)
    prices = np.array(all_prices)
    atrs = df_4h['atr'].values[-len(prices):]
    atrs = np.nan_to_num(atrs, nan=np.nanmean(df_4h['atr'].values))
    
    # Split: 60% train, 20% val, 20% test
    n = len(embeddings)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    GLOBAL_DATA = {
        'train': {
            'embeddings': embeddings[:train_end],
            'pred_4h': pred_4h[:train_end],
            'pred_12h': pred_12h[:train_end],
            'pred_24h': pred_24h[:train_end],
            'regimes': regimes[:train_end],
            'prices': prices[:train_end],
            'atrs': atrs[:train_end]
        },
        'val': {
            'embeddings': embeddings[train_end:val_end],
            'pred_4h': pred_4h[train_end:val_end],
            'pred_12h': pred_12h[train_end:val_end],
            'pred_24h': pred_24h[train_end:val_end],
            'regimes': regimes[train_end:val_end],
            'prices': prices[train_end:val_end],
            'atrs': atrs[train_end:val_end]
        },
        'test': {
            'embeddings': embeddings[val_end:],
            'pred_4h': pred_4h[val_end:],
            'pred_12h': pred_12h[val_end:],
            'pred_24h': pred_24h[val_end:],
            'regimes': regimes[val_end:],
            'prices': prices[val_end:],
            'atrs': atrs[val_end:]
        },
        'device': device
    }
    
    logger.info(f"   Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")
    
    return GLOBAL_DATA


def evaluate_config(config: AdvancedTradingConfig, data: dict, device: str, n_episodes: int = 5) -> dict:
    """Evalúa una configuración específica."""
    
    env = AdvancedTradingEnv(
        data['embeddings'],
        data['pred_4h'],
        data['pred_12h'],
        data['pred_24h'],
        data['regimes'],
        data['prices'],
        data['atrs'],
        config
    )
    
    agent = SACAdvanced(
        state_dim=env.state_dim,
        hidden_dim=256,
        device=device
    )
    
    # Entrenamiento rápido
    agent.train(
        env=env,
        total_steps=50000,  # Reducido para optimización
        batch_size=256,
        start_steps=2000,
        log_interval=100000  # Sin logs
    )
    
    # Evaluar
    returns = []
    drawdowns = []
    exposures = []
    
    for _ in range(n_episodes):
        state, regime = env.reset()
        done = False
        positions = []
        
        while not done:
            action = agent.select_action(state, regime, deterministic=True)
            state, _, done, regime, info = env.step(action)
            positions.append(abs(info['position'] * info['position_size']))
        
        returns.append(info['total_return'])
        drawdowns.append(info['drawdown'])
        exposures.append(np.mean(np.array(positions) > 0.1))
    
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-8
    sharpe = mean_return / std_return * np.sqrt(252 * 6)  # Anualizado
    
    return {
        'sharpe': sharpe,
        'return': mean_return,
        'drawdown': np.mean(drawdowns),
        'exposure': np.mean(exposures)
    }


def objective(trial: optuna.Trial) -> float:
    """Función objetivo para Optuna."""
    
    data = load_data_once()
    
    # Hiperparámetros a optimizar
    config = AdvancedTradingConfig(
        initial_capital=10000.0,
        max_leverage=trial.suggest_int('max_leverage', 2, 5),
        kelly_fraction=trial.suggest_float('kelly_fraction', 0.3, 1.0),
        atr_multiplier_stop=trial.suggest_float('atr_stop', 1.5, 3.5),
        atr_multiplier_trail=trial.suggest_float('atr_trail', 1.0, 2.5),
        max_drawdown=trial.suggest_float('max_drawdown', 0.10, 0.25),
        min_position=trial.suggest_float('min_position', 0.05, 0.2),
        max_position=trial.suggest_float('max_position', 0.8, 1.0),
    )
    
    # Evaluar en validación
    try:
        metrics = evaluate_config(config, data['val'], data['device'])
        
        # Objetivo: maximizar Sharpe, pero penalizar drawdown excesivo
        score = metrics['sharpe']
        
        # Penalizar si exposure muy bajo (muy conservador)
        if metrics['exposure'] < 0.2:
            score -= 0.5
        
        # Penalizar drawdown alto
        if metrics['drawdown'] > 0.15:
            score -= 1.0
        
        # Reportar para pruning
        trial.report(score, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
        
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return -10.0


def main():
    print("\n" + "=" * 70)
    print("          🔧 OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
    print("=" * 70 + "\n")
    
    # Cargar datos
    load_data_once()
    
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        study_name='trading_optimization'
    )
    
    # Optimizar
    logger.info("🚀 Iniciando optimización (50 trials)...")
    logger.info("   Esto puede tardar 1-2 horas...\n")
    
    study.optimize(
        objective,
        n_trials=50,
        show_progress_bar=True,
        n_jobs=1  # Secuencial para GPU
    )
    
    # Mejores hiperparámetros
    print("\n" + "=" * 70)
    print("                    ✅ OPTIMIZACIÓN COMPLETADA")
    print("=" * 70)
    
    print(f"\n📊 Mejor Sharpe encontrado: {study.best_value:.2f}")
    print(f"\n🎯 Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    # Evaluar en TEST con mejores parámetros
    print("\n🔬 Evaluando en TEST SET con mejores parámetros...")
    
    data = load_data_once()
    
    best_config = AdvancedTradingConfig(
        initial_capital=10000.0,
        max_leverage=study.best_params['max_leverage'],
        kelly_fraction=study.best_params['kelly_fraction'],
        atr_multiplier_stop=study.best_params['atr_stop'],
        atr_multiplier_trail=study.best_params['atr_trail'],
        max_drawdown=study.best_params['max_drawdown'],
        min_position=study.best_params['min_position'],
        max_position=study.best_params['max_position'],
    )
    
    test_metrics = evaluate_config(best_config, data['test'], data['device'], n_episodes=10)
    
    print(f"\n📈 RESULTADOS EN TEST SET:")
    print(f"   Sharpe Ratio: {test_metrics['sharpe']:.2f}")
    print(f"   Return: {test_metrics['return']*100:+.2f}%")
    print(f"   Max Drawdown: {test_metrics['drawdown']*100:.2f}%")
    print(f"   Exposure: {test_metrics['exposure']*100:.1f}%")
    
    # Guardar mejores parámetros
    params_path = root_dir / "artifacts/best_params.npz"
    np.savez(params_path, **study.best_params)
    logger.info(f"\n💾 Parámetros guardados: {params_path}")
    
    # Guardar estudio
    study_path = root_dir / "artifacts/optuna_study.pkl"
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    logger.info(f"💾 Estudio guardado: {study_path}")
    
    print(f"\n🎯 Siguiente paso: Entrenar modelo final con parámetros optimizados")
    print(f"   python scripts/06_train_final.py")


if __name__ == "__main__":
    main()
