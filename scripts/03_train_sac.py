"""
Script 03: Entrena el Agente SAC con el Detector de Patrones.
Combina el embedding del LSTM con SAC para aprender a operar.

Uso:
    python scripts/03_train_sac.py
"""

import sys
from pathlib import Path

# Añadir raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
import numpy as np
import pandas as pd
from loguru import logger

from src.features.indicators import calculate_all_indicators
from src.models.pattern_detector import PatternDetectorLSTM, PatternDataset
from src.models.sac_agent import SACAgent, TradingEnvironment, TradingConfig


def load_pattern_detector(artifacts_path: Path, device: str) -> tuple:
    """Carga el detector de patrones entrenado."""
    
    # Cargar checkpoint
    checkpoint_path = artifacts_path / "pattern_detector/pattern_detector.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['model_config']
    
    # Recrear modelo
    model = PatternDetectorLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        embedding_dim=config['embedding_dim'],
        bidirectional=config['bidirectional']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    # Cargar predictor head
    predictor = torch.nn.Sequential(
        torch.nn.Linear(config['embedding_dim'], 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 3)
    ).to(device)
    predictor.load_state_dict(checkpoint['predictor_state'])
    predictor.eval()
    
    logger.info(f"📂 Detector de patrones cargado: {checkpoint_path}")
    
    return model, predictor, config


def generate_embeddings(
    model: PatternDetectorLSTM,
    predictor: torch.nn.Module,
    df: pd.DataFrame,
    seq_length: int,
    device: str
) -> tuple:
    """
    Genera embeddings y predicciones para todo el dataset.
    """
    logger.info("🧠 Generando embeddings y predicciones...")
    
    # Crear dataset
    dataset = PatternDataset(df, seq_length=seq_length, horizon=4)
    
    # Cargar normalizer
    norm_path = root_dir / "artifacts/scaler/pattern_normalizer.npz"
    norm_data = np.load(norm_path, allow_pickle=True)
    
    embeddings_list = []
    predictions_list = []
    
    model.eval()
    predictor.eval()
    
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            # Obtener embedding
            embedding, _ = model(batch_x)
            
            # Obtener predicciones
            pred = predictor(embedding)
            
            embeddings_list.append(embedding.cpu().numpy())
            predictions_list.append(pred.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    predictions = np.vstack(predictions_list)
    
    logger.info(f"   Embeddings shape: {embeddings.shape}")
    logger.info(f"   Predictions shape: {predictions.shape}")
    
    return embeddings, predictions, dataset.valid_indices


def main():
    """Entrena el agente SAC."""
    
    print("\n" + "=" * 60)
    print("   🤖 ENTRENAMIENTO AGENTE SAC")
    print("=" * 60 + "\n")
    
    # ========== CONFIGURACIÓN ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Device: {device}")
    
    config = {
        # Pattern Detector
        'seq_length': 50,
        
        # SAC
        'hidden_dim': 256,
        'lr': 1e-4,          # Reducido para más estabilidad
        'gamma': 0.99,
        'tau': 0.005,
        
        # Training - MÁS ENTRENAMIENTO
        'total_steps': 500000,    # 500k steps (antes 200k)
        'batch_size': 256,
        'start_steps': 10000,     # Más exploración inicial
        'eval_interval': 25000,
        'log_interval': 10000,
        
        # Trading
        'initial_capital': 10000.0,
        'leverage': 3,
        'taker_fee': 0.0006,
    }
    
    logger.info(f"📋 Configuración:")
    for k, v in config.items():
        logger.info(f"   {k}: {v}")
    
    # ========== CARGAR DATOS ==========
    logger.info("\n📥 Cargando datos...")
    
    data_path = root_dir / "data/raw/btcusdt_1h.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"   Filas: {len(df):,}")
    
    # Calcular indicadores
    df = calculate_all_indicators(df)
    logger.info(f"   Filas con indicadores: {len(df):,}")
    
    # ========== CARGAR PATTERN DETECTOR ==========
    logger.info("\n🧠 Cargando detector de patrones...")
    
    artifacts_path = root_dir / "artifacts"
    model, predictor, model_config = load_pattern_detector(artifacts_path, device)
    
    # ========== GENERAR EMBEDDINGS ==========
    embeddings, predictions, valid_indices = generate_embeddings(
        model, predictor, df, config['seq_length'], device
    )
    
    # Obtener precios correspondientes
    prices = df['close'].values[valid_indices]
    
    logger.info(f"   Prices shape: {prices.shape}")
    
    # ========== SPLIT TRAIN/TEST ==========
    logger.info("\n✂️ Dividiendo datos (80% train / 20% test)...")
    
    split_idx = int(len(embeddings) * 0.8)
    
    train_embeddings = embeddings[:split_idx]
    train_predictions = predictions[:split_idx]
    train_prices = prices[:split_idx]
    
    test_embeddings = embeddings[split_idx:]
    test_predictions = predictions[split_idx:]
    test_prices = prices[split_idx:]
    
    logger.info(f"   Train: {len(train_embeddings):,} samples")
    logger.info(f"   Test: {len(test_embeddings):,} samples")
    
    # ========== CREAR ENTORNOS ==========
    logger.info("\n🎮 Creando entornos de trading...")
    
    trading_config = TradingConfig(
        initial_capital=config['initial_capital'],
        leverage=config['leverage'],
        taker_fee=config['taker_fee']
    )
    
    train_env = TradingEnvironment(
        train_embeddings, train_predictions, train_prices, trading_config
    )
    
    test_env = TradingEnvironment(
        test_embeddings, test_predictions, test_prices, trading_config
    )
    
    logger.info(f"   Train env steps: {train_env.n_steps}")
    logger.info(f"   Test env steps: {test_env.n_steps}")
    logger.info(f"   State dim: {train_env.state_dim}")
    
    # ========== CREAR AGENTE SAC ==========
    logger.info("\n🤖 Creando agente SAC...")
    
    agent = SACAgent(
        state_dim=train_env.state_dim,
        hidden_dim=config['hidden_dim'],
        lr=config['lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        device=device
    )
    
    # ========== ENTRENAR ==========
    logger.info("\n🏋️ Iniciando entrenamiento...")
    
    history = agent.train(
        env=train_env,
        total_steps=config['total_steps'],
        batch_size=config['batch_size'],
        start_steps=config['start_steps'],
        log_interval=config['log_interval'],
        eval_interval=config['eval_interval']
    )
    
    # ========== EVALUAR EN TEST ==========
    logger.info("\n📊 Evaluación final en test set...")
    
    test_returns = []
    test_trades = []
    test_drawdowns = []
    
    for episode in range(10):
        state = test_env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, _, done, info = test_env.step(action)
        
        test_returns.append(info['total_return'])
        test_trades.append(info['total_trades'])
        test_drawdowns.append(info['drawdown'])
    
    mean_return = np.mean(test_returns)
    std_return = np.std(test_returns)
    mean_trades = np.mean(test_trades)
    max_drawdown = np.max(test_drawdowns)
    
    # Calcular Sharpe aproximado
    sharpe = mean_return / (std_return + 1e-8) * np.sqrt(252)  # Anualizado
    
    # ========== GUARDAR MODELO ==========
    model_path = root_dir / "artifacts/sac/sac_agent.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_path))
    
    # Guardar configuración
    config_path = root_dir / "artifacts/sac/config.npz"
    np.savez(config_path, **{k: str(v) for k, v in config.items()})
    
    # ========== RESUMEN FINAL ==========
    print("\n" + "=" * 60)
    print("   ✅ ENTRENAMIENTO SAC COMPLETADO")
    print("=" * 60)
    
    print(f"\n📊 Resultados en TEST SET (10 episodios):")
    print(f"   Retorno medio: {mean_return*100:+.2f}%")
    print(f"   Retorno std: {std_return*100:.2f}%")
    print(f"   Sharpe ratio: {sharpe:.2f}")
    print(f"   Max drawdown: {max_drawdown*100:.2f}%")
    print(f"   Trades promedio: {mean_trades:.1f}")
    
    print(f"\n📈 Retornos por episodio:")
    for i, ret in enumerate(test_returns):
        emoji = "🟢" if ret > 0 else "🔴"
        print(f"   {emoji} Episodio {i+1}: {ret*100:+.2f}%")
    
    print(f"\n💾 Artefactos guardados:")
    print(f"   Modelo: {model_path}")
    print(f"   Config: {config_path}")
    
    # Guardar historial
    history_path = root_dir / "artifacts/sac/training_history.npz"
    np.savez(
        history_path,
        rewards=history['rewards'],
        q_loss=history['q_loss'],
        policy_loss=history['policy_loss'],
        eval_returns=history['eval_returns']
    )
    
    print(f"\n🎉 ¡Agente SAC listo!")
    print(f"   Siguiente paso: python scripts/04_backtest.py")


if __name__ == "__main__":
    main()
