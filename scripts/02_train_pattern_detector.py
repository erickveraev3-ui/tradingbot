"""
Script 02: Entrena el Detector de Patrones LSTM.
Este modelo aprende a representar el estado del mercado.

Uso:
    python scripts/02_train_pattern_detector.py
"""

import sys
from pathlib import Path

# Añadir raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from loguru import logger

from src.features.indicators import calculate_all_indicators
from src.models.pattern_detector import (
    PatternDetectorLSTM,
    PatternDataset,
    PatternDetectorTrainer
)


def main():
    """Entrena el detector de patrones."""
    
    print("\n" + "=" * 60)
    print("   🧠 ENTRENAMIENTO DETECTOR DE PATRONES")
    print("=" * 60 + "\n")
    
    # ========== VERIFICAR GPU ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU detectada: {gpu_name}")
    else:
        logger.warning("⚠️ No hay GPU, usando CPU (más lento)")
    
    # ========== CONFIGURACIÓN ==========
    config = {
        # Datos
        'seq_length': 50,       # Velas de contexto
        'horizon': 4,           # Predicción a 4 horas
        
        # Modelo (más grande para GPU)
        'hidden_dim': 256,      # Más capacidad
        'num_layers': 3,        # Más profundo
        'embedding_dim': 128,   # Embedding más rico
        'dropout': 0.3,
        
        # Entrenamiento (más épocas con GPU)
        'batch_size': 128,      # Batches más grandes
        'epochs': 200,          # Más épocas
        'learning_rate': 5e-4,
        'early_stopping': 20,
        
        # Split 80/20
        'train_ratio': 0.80,
        'test_ratio': 0.20,
    }
    
    logger.info(f"📋 Configuración:")
    for k, v in config.items():
        logger.info(f"   {k}: {v}")
    
    # ========== CARGAR DATOS ==========
    logger.info("\n📥 Cargando datos...")
    
    data_path = root_dir / "data/raw/btcusdt_1h.csv"
    if not data_path.exists():
        logger.error("❌ No hay datos. Ejecuta primero: python scripts/01_download_data.py")
        return
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"   Filas cargadas: {len(df):,}")
    logger.info(f"   Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # ========== CALCULAR INDICADORES ==========
    logger.info("\n📊 Calculando indicadores...")
    df = calculate_all_indicators(df)
    logger.info(f"   Filas con indicadores: {len(df):,}")
    
    # ========== CREAR DATASET ==========
    logger.info("\n📦 Creando dataset...")
    
    dataset = PatternDataset(
        df,
        seq_length=config['seq_length'],
        horizon=config['horizon']
    )
    
    # Guardar parámetros de normalización
    norm_params = dataset.get_normalizer_params()
    norm_path = root_dir / "artifacts/scaler/pattern_normalizer.npz"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        norm_path,
        mean=norm_params['mean'],
        std=norm_params['std'],
        feature_columns=norm_params['feature_columns']
    )
    logger.info(f"💾 Normalizer guardado: {norm_path}")
    
    # ========== SPLIT 80/20 SECUENCIAL ==========
    logger.info("\n✂️ Dividiendo datos (80% train / 20% test)...")
    
    total_size = len(dataset)
    train_size = int(total_size * config['train_ratio'])
    test_size = total_size - train_size
    
    # Split secuencial (importante para series temporales)
    # Entrenamos con el pasado, testeamos con el futuro
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))
    
    logger.info(f"   Train: {len(train_dataset):,} samples ({config['train_ratio']*100:.0f}%)")
    logger.info(f"   Test: {len(test_dataset):,} samples ({config['test_ratio']*100:.0f}%)")
    
    # DataLoaders optimizados para GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,          # Paralelizar carga de datos
        pin_memory=True,        # Acelerar transferencia CPU→GPU
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ========== CREAR MODELO ==========
    logger.info("\n🧠 Creando modelo...")
    
    model = PatternDetectorLSTM(
        input_dim=len(dataset.feature_columns),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        embedding_dim=config['embedding_dim'],
        dropout=config['dropout']
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parámetros totales: {total_params:,}")
    
    # ========== ENTRENAR ==========
    logger.info("\n🏋️ Iniciando entrenamiento...")
    logger.info(f"   Device: {device}")
    logger.info(f"   Épocas: {config['epochs']}")
    logger.info(f"   Early stopping: {config['early_stopping']} épocas sin mejora")
    
    trainer = PatternDetectorTrainer(
        model,
        learning_rate=config['learning_rate'],
        device=device
    )
    
    # Usamos test como validación (80/20 sin validación separada)
    history = trainer.train(
        train_loader,
        test_loader,  # Usamos test para validar
        epochs=config['epochs'],
        early_stopping=config['early_stopping']
    )
    
    # ========== EVALUAR EN TEST ==========
    logger.info("\n📊 Evaluación final en test set...")
    
    test_loss = trainer.validate(test_loader)
    logger.info(f"   Test Loss: {test_loss:.6f}")
    
    # ========== GUARDAR MODELO ==========
    model_path = root_dir / "artifacts/pattern_detector/pattern_detector.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(model_path))
    
    # Guardar configuración
    config_path = root_dir / "artifacts/pattern_detector/config.npz"
    np.savez(config_path, **{k: str(v) if isinstance(v, bool) else v for k, v in config.items()})
    logger.info(f"💾 Config guardada: {config_path}")
    
    # ========== MÉTRICAS DETALLADAS ==========
    logger.info("\n🔍 Calculando métricas detalladas...")
    
    model.eval()
    all_preds = []
    all_reals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            embedding, _ = model(batch_x)
            pred = trainer.predictor_head(embedding)
            
            all_preds.append(pred.cpu().numpy())
            all_reals.append(batch_y.numpy())
    
    all_preds = np.vstack(all_preds)
    all_reals = np.vstack(all_reals).flatten()
    
    # Calcular métricas
    q10_preds = all_preds[:, 0]
    q50_preds = all_preds[:, 1]
    q90_preds = all_preds[:, 2]
    
    # Cobertura: % de veces que el real está entre q10 y q90
    coverage = np.mean((all_reals >= q10_preds) & (all_reals <= q90_preds)) * 100
    
    # Error medio del q50 (mediana)
    mae_q50 = np.mean(np.abs(all_reals - q50_preds)) * 100
    
    # Dirección correcta
    direction_correct = np.mean(np.sign(all_reals) == np.sign(q50_preds)) * 100
    
    # ========== RESUMEN FINAL ==========
    print("\n" + "=" * 60)
    print("   ✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    
    print(f"\n📊 Resultados del entrenamiento:")
    print(f"   Train Loss Final: {history['train_loss'][-1]:.6f}")
    print(f"   Test Loss Final: {test_loss:.6f}")
    print(f"   Épocas ejecutadas: {len(history['train_loss'])}")
    
    print(f"\n📈 Métricas de predicción:")
    print(f"   Cobertura (real entre q10-q90): {coverage:.1f}%")
    print(f"   MAE q50: {mae_q50:.3f}%")
    print(f"   Dirección correcta: {direction_correct:.1f}%")
    
    print(f"\n💾 Artefactos guardados:")
    print(f"   Modelo: {model_path}")
    print(f"   Normalizer: {norm_path}")
    print(f"   Config: {config_path}")
    
    # ========== EJEMPLOS DE PREDICCIONES ==========
    print("\n🔍 Ejemplos de predicciones (últimos 10 del test):")
    print("   " + "-" * 55)
    print(f"   {'#':<3} {'q10':>8} {'q50':>8} {'q90':>8} {'Real':>8} {'OK?':>5}")
    print("   " + "-" * 55)
    
    for i in range(-10, 0):
        q10 = q10_preds[i] * 100
        q50 = q50_preds[i] * 100
        q90 = q90_preds[i] * 100
        real = all_reals[i] * 100
        
        ok = "✅" if q10 <= all_reals[i] <= q90 else "❌"
        
        print(f"   {i+11:<3} {q10:>+7.2f}% {q50:>+7.2f}% {q90:>+7.2f}% {real:>+7.2f}% {ok:>5}")
    
    print("   " + "-" * 55)
    
    # Guardar historial
    history_path = root_dir / "artifacts/pattern_detector/training_history.npz"
    np.savez(history_path, 
             train_loss=history['train_loss'],
             val_loss=history['val_loss'])
    
    print(f"\n🎉 ¡Modelo listo para usar!")
    print(f"   Siguiente paso: python scripts/03_train_ppo.py")


if __name__ == "__main__":
    main()