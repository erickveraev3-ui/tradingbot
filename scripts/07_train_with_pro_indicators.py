"""
Script 07: Entrenamiento con Indicadores Profesionales.

Mejoras:
- 49 indicadores PRO (vs 14 básicos)
- Kelly más conservador
- Regularización aumentada
- Más trials de Optuna

Uso:
    python scripts/07_train_with_pro_indicators.py
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import torch
import numpy as np
import pandas as pd
import optuna
from torch.utils.data import DataLoader, Dataset
from loguru import logger
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.features.indicators import calculate_all_indicators
from src.features.indicators_pro import calculate_pro_indicators, get_pro_feature_columns
from src.models.contrastive_pretrain import (
    ContrastiveEncoder, ContrastiveConfig,
    ContrastiveDataset, ContrastiveTrainer
)
from src.models.sac_advanced import SACAdvanced, AdvancedTradingEnv, AdvancedTradingConfig


class ProDataset(Dataset):
    """Dataset con indicadores profesionales."""
    
    def __init__(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        seq_len_1h: int = 24,
        seq_len_4h: int = 50,
        seq_len_1d: int = 30
    ):
        self.seq_len_1h = seq_len_1h
        self.seq_len_4h = seq_len_4h
        self.seq_len_1d = seq_len_1d
        
        # Feature columns combinadas (básicas + PRO)
        self.basic_features = [
            'return_1', 'return_4', 'return_24',
            'rsi', 'macd_hist', 'bb_position', 'atr_pct',
            'adx', 'di_plus', 'di_minus',
            'vol_ratio', 'trend_ema', 'trend_long', 'trend_strength'
        ]
        
        self.pro_features = get_pro_feature_columns()
        self.all_features = self.basic_features + self.pro_features
        
        # Preparar datos
        logger.info("📊 Preparando datos con indicadores PRO...")
        self.data_1h, self.feat_cols_1h = self._prepare(df_1h)
        self.data_4h, self.feat_cols_4h = self._prepare(df_4h)
        self.data_1d, self.feat_cols_1d = self._prepare(df_1d)
        
        logger.info(f"   Features 1H: {len(self.feat_cols_1h)}")
        logger.info(f"   Features 4H: {len(self.feat_cols_4h)}")
        logger.info(f"   Features 1D: {len(self.feat_cols_1d)}")
        
        # Timestamps y precios
        self.ts_1h = pd.to_datetime(df_1h['timestamp']).values
        self.ts_4h = pd.to_datetime(df_4h['timestamp']).values
        self.ts_1d = pd.to_datetime(df_1d['timestamp']).values
        
        self.prices_4h = df_4h['close'].values
        
        # ATR
        self.atrs = df_4h['atr'].values if 'atr' in df_4h.columns else self._calc_atr(df_4h)
        
        # Targets
        self.returns_4h = self._future_returns(self.prices_4h, 1)
        self.returns_12h = self._future_returns(self.prices_4h, 3)
        self.returns_24h = self._future_returns(self.prices_4h, 6)
        
        # Régimen
        self.regimes = self._calc_regimes(df_4h)
        
        # Índices válidos
        self._calc_valid_indices()
        
        logger.info(f"✅ ProDataset: {len(self.valid_indices)} samples")
    
    def _prepare(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        available = [c for c in self.all_features if c in df.columns]
        data = df[available].values.astype(np.float32)
        
        # Normalización robusta (menos sensible a outliers)
        median = np.nanmedian(data, axis=0)
        mad = np.nanmedian(np.abs(data - median), axis=0) + 1e-8
        data = (data - median) / (mad * 1.4826)  # MAD to std conversion
        
        # Clip outliers
        data = np.clip(data, -5, 5)
        data = np.nan_to_num(data, 0)
        
        return data, available
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        tr = np.insert(tr, 0, high[0] - low[0])
        atr = pd.Series(tr).rolling(period).mean().values
        return np.nan_to_num(atr, nan=np.nanmean(tr))
    
    def _future_returns(self, prices: np.ndarray, horizon: int) -> np.ndarray:
        returns = np.zeros(len(prices))
        for i in range(len(prices) - horizon):
            returns[i] = (prices[i + horizon] - prices[i]) / prices[i]
        return returns.astype(np.float32)
    
    def _calc_regimes(self, df: pd.DataFrame) -> np.ndarray:
        regimes = np.zeros(len(df), dtype=np.int64)
        returns = df['close'].pct_change(20).values
        volatility = df['close'].pct_change().rolling(20).std().values
        adx = df['adx'].values if 'adx' in df.columns else np.zeros(len(df))
        
        vol_threshold = np.nanpercentile(volatility, 75)
        
        for i in range(len(df)):
            if np.isnan(returns[i]) or np.isnan(volatility[i]):
                regimes[i] = 2
                continue
            if volatility[i] > vol_threshold:
                regimes[i] = 3
            elif returns[i] > 0.02 and adx[i] > 25:
                regimes[i] = 0
            elif returns[i] < -0.02 and adx[i] > 25:
                regimes[i] = 1
            else:
                regimes[i] = 2
        return regimes
    
    def _calc_valid_indices(self):
        self.valid_indices = []
        self.alignment = []
        
        for idx_4h in range(self.seq_len_4h, len(self.data_4h) - 6):
            ts = self.ts_4h[idx_4h]
            
            mask_1d = self.ts_1d <= ts
            if mask_1d.sum() < self.seq_len_1d:
                continue
            idx_1d = mask_1d.sum() - 1
            
            mask_1h = self.ts_1h <= ts
            if mask_1h.sum() < self.seq_len_1h:
                continue
            idx_1h = mask_1h.sum() - 1
            
            self.valid_indices.append(idx_4h)
            self.alignment.append((idx_1d, idx_1h))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        idx_4h = self.valid_indices[idx]
        idx_1d, idx_1h = self.alignment[idx]
        
        x_1h = self.data_1h[idx_1h - self.seq_len_1h + 1:idx_1h + 1]
        x_4h = self.data_4h[idx_4h - self.seq_len_4h + 1:idx_4h + 1]
        x_1d = self.data_1d[idx_1d - self.seq_len_1d + 1:idx_1d + 1]
        
        return {
            'x_1h': torch.tensor(x_1h, dtype=torch.float32),
            'x_4h': torch.tensor(x_4h, dtype=torch.float32),
            'x_1d': torch.tensor(x_1d, dtype=torch.float32),
            'returns': torch.tensor([
                self.returns_4h[idx_4h],
                self.returns_12h[idx_4h],
                self.returns_24h[idx_4h]
            ], dtype=torch.float32),
            'regime': torch.tensor(self.regimes[idx_4h], dtype=torch.long),
            'price': torch.tensor(self.prices_4h[idx_4h], dtype=torch.float32),
            'atr': torch.tensor(self.atrs[idx_4h], dtype=torch.float32)
        }
    
    @property
    def input_dim(self):
        return self.data_4h.shape[1]


class ProFusionNetwork(torch.nn.Module):
    """Red de fusión mejorada con regularización."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        
        # LSTM por timeframe con más regularización
        self.lstm_1h = torch.nn.LSTM(input_dim, hidden_dim // 2, num_layers=2, 
                                      batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm_4h = torch.nn.LSTM(input_dim, hidden_dim // 2, num_layers=2,
                                      batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm_1d = torch.nn.LSTM(input_dim, hidden_dim // 2, num_layers=2,
                                      batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fusion con dropout alto
        fusion_input = hidden_dim * 3
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(fusion_input, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        
        # Heads
        self.regime_head = torch.nn.Linear(hidden_dim, 4)
        self.return_head = torch.nn.Linear(hidden_dim, 9)  # 3 horizontes x 3 cuantiles
        self.embedding_head = torch.nn.Linear(hidden_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, x_1h, x_4h, x_1d):
        # LSTM embeddings
        _, (h_1h, _) = self.lstm_1h(x_1h)
        _, (h_4h, _) = self.lstm_4h(x_4h)
        _, (h_1d, _) = self.lstm_1d(x_1d)
        
        # Concatenar hidden states
        emb_1h = torch.cat([h_1h[-2], h_1h[-1]], dim=-1)
        emb_4h = torch.cat([h_4h[-2], h_4h[-1]], dim=-1)
        emb_1d = torch.cat([h_1d[-2], h_1d[-1]], dim=-1)
        
        combined = torch.cat([emb_1h, emb_4h, emb_1d], dim=-1)
        fused = self.fusion(combined)
        
        return {
            'embedding': self.embedding_head(fused),
            'regime_logits': self.regime_head(fused),
            'return_preds': self.return_head(fused).view(-1, 3, 3)
        }


def train_fusion(model, train_loader, val_loader, epochs: int = 100, device: str = 'cuda'):
    """Entrena red de fusión con early stopping."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_1h = batch['x_1h'].to(device)
            x_4h = batch['x_4h'].to(device)
            x_1d = batch['x_1d'].to(device)
            returns = batch['returns'].to(device)
            regime = batch['regime'].to(device)
            
            out = model(x_1h, x_4h, x_1d)
            
            # Loss
            loss_ret = torch.nn.functional.mse_loss(out['return_preds'][:, :, 1], returns)
            loss_regime = torch.nn.functional.cross_entropy(out['regime_logits'], regime)
            loss = loss_ret + 0.3 * loss_regime
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_1h = batch['x_1h'].to(device)
                x_4h = batch['x_4h'].to(device)
                x_1d = batch['x_1d'].to(device)
                returns = batch['returns'].to(device)
                
                out = model(x_1h, x_4h, x_1d)
                loss = torch.nn.functional.mse_loss(out['return_preds'][:, :, 1], returns)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Época {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if patience_counter >= max_patience:
            logger.info(f"Early stopping en época {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model


def generate_embeddings(model, dataset, device):
    """Genera embeddings del modelo de fusión."""
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    embeddings, pred_4h, pred_12h, pred_24h = [], [], [], []
    regimes, prices, atrs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            x_1h = batch['x_1h'].to(device)
            x_4h = batch['x_4h'].to(device)
            x_1d = batch['x_1d'].to(device)
            
            out = model(x_1h, x_4h, x_1d)
            
            embeddings.append(out['embedding'].cpu().numpy())
            preds = out['return_preds'].cpu().numpy()
            pred_4h.append(preds[:, 0, :])
            pred_12h.append(preds[:, 1, :])
            pred_24h.append(preds[:, 2, :])
            regimes.append(out['regime_logits'].argmax(dim=-1).cpu().numpy())
            prices.append(batch['price'].numpy())
            atrs.append(batch['atr'].numpy())
    
    return {
        'embeddings': np.vstack(embeddings),
        'pred_4h': np.vstack(pred_4h),
        'pred_12h': np.vstack(pred_12h),
        'pred_24h': np.vstack(pred_24h),
        'regimes': np.concatenate(regimes),
        'prices': np.concatenate(prices),
        'atrs': np.concatenate(atrs)
    }


def train_sac_optuna(train_data, val_data, n_trials: int = 75, device: str = 'cuda'):
    """Optimización SAC con hiperparámetros más conservadores."""
    
    def objective(trial):
        # Hiperparámetros MÁS CONSERVADORES
        config = AdvancedTradingConfig(
            initial_capital=10000.0,
            max_leverage=trial.suggest_int('max_leverage', 2, 3),  # Reducido
            kelly_fraction=trial.suggest_float('kelly_fraction', 0.2, 0.5),  # Más conservador
            atr_multiplier_stop=trial.suggest_float('atr_stop', 2.0, 3.0),
            atr_multiplier_trail=trial.suggest_float('atr_trail', 1.5, 2.5),
            max_drawdown=trial.suggest_float('max_drawdown', 0.08, 0.15),  # Más estricto
            min_position=trial.suggest_float('min_position', 0.05, 0.15),
            max_position=trial.suggest_float('max_position', 0.5, 0.8),  # Reducido
        )
        
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256])
        lr = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
        gamma = trial.suggest_float('gamma', 0.98, 0.999)
        tau = trial.suggest_float('tau', 0.002, 0.01)
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        
        env = AdvancedTradingEnv(
            val_data['embeddings'], val_data['pred_4h'], val_data['pred_12h'],
            val_data['pred_24h'], val_data['regimes'], val_data['prices'],
            val_data['atrs'], config
        )
        
        agent = SACAdvanced(
            state_dim=env.state_dim, hidden_dim=hidden_dim,
            lr=lr, gamma=gamma, tau=tau, device=device
        )
        
        try:
            agent.train(env=env, total_steps=30000, batch_size=batch_size,
                       start_steps=2000, log_interval=100000)
            
            metrics = agent.evaluate(env, n_episodes=5)
            
            # Score: Sharpe + bonus por retorno positivo - penalización por drawdown
            score = metrics['sharpe']
            
            if metrics['mean_return'] > 0:
                score += 1.0
            if metrics['mean_return'] < -0.05:
                score -= 2.0
            if metrics['max_drawdown'] > 0.12:
                score -= 1.0
            
            return score
            
        except Exception as e:
            return -10.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study


def main():
    print("\n" + "=" * 70)
    print("    🚀 ENTRENAMIENTO CON INDICADORES PROFESIONALES")
    print("=" * 70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Device: {device}")
    
    # =========================================
    # CARGAR Y PREPARAR DATOS
    # =========================================
    logger.info("\n📥 Cargando datos...")
    
    df_1h = pd.read_csv(root_dir / "data/raw/btcusdt_1h.csv")
    df_4h = pd.read_csv(root_dir / "data/raw/btcusdt_4h.csv")
    df_1d = pd.read_csv(root_dir / "data/raw/btcusdt_1d.csv")
    
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
    
    # Calcular TODOS los indicadores
    logger.info("📊 Calculando indicadores básicos...")
    df_1h = calculate_all_indicators(df_1h)
    df_4h = calculate_all_indicators(df_4h)
    df_1d = calculate_all_indicators(df_1d)
    
    logger.info("📊 Calculando indicadores PRO...")
    df_1h = calculate_pro_indicators(df_1h)
    df_4h = calculate_pro_indicators(df_4h)
    df_1d = calculate_pro_indicators(df_1d)
    
    # =========================================
    # CREAR DATASET
    # =========================================
    logger.info("\n📦 Creando dataset PRO...")
    
    full_dataset = ProDataset(df_1h, df_4h, df_1d)
    
    # Split: 50% train, 20% val, 15% test, 15% holdout
    n = len(full_dataset)
    train_end = int(n * 0.50)
    val_end = int(n * 0.70)
    test_end = int(n * 0.85)
    
    train_idx = list(range(train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, test_end))
    holdout_idx = list(range(test_end, n))
    
    logger.info(f"   Train: {len(train_idx)}")
    logger.info(f"   Val: {len(val_idx)}")
    logger.info(f"   Test: {len(test_idx)}")
    logger.info(f"   Holdout: {len(holdout_idx)} (INTOCABLE)")
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # =========================================
    # FASE 1: ENTRENAR RED DE FUSIÓN
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 1: ENTRENANDO RED DE FUSIÓN PRO")
    logger.info("=" * 60)
    
    fusion_model = ProFusionNetwork(
        input_dim=full_dataset.input_dim,
        hidden_dim=256,
        output_dim=128,
        dropout=0.3  # Alta regularización
    )
    
    fusion_model = train_fusion(fusion_model, train_loader, val_loader, epochs=100, device=device)
    
    # Guardar
    (root_dir / "artifacts/fusion_pro").mkdir(parents=True, exist_ok=True)
    torch.save(fusion_model.state_dict(), root_dir / "artifacts/fusion_pro/model.pt")
    
    # =========================================
    # FASE 2: GENERAR EMBEDDINGS
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 2: GENERANDO EMBEDDINGS")
    logger.info("=" * 60)
    
    train_sub = torch.utils.data.Subset(full_dataset, train_idx)
    val_sub = torch.utils.data.Subset(full_dataset, val_idx)
    test_sub = torch.utils.data.Subset(full_dataset, test_idx)
    holdout_sub = torch.utils.data.Subset(full_dataset, holdout_idx)
    
    train_data = generate_embeddings(fusion_model, train_sub, device)
    val_data = generate_embeddings(fusion_model, val_sub, device)
    test_data = generate_embeddings(fusion_model, test_sub, device)
    holdout_data = generate_embeddings(fusion_model, holdout_sub, device)
    
    logger.info(f"   Input dim: {full_dataset.input_dim}")
    logger.info(f"   Embedding dim: {train_data['embeddings'].shape[1]}")
    
    # =========================================
    # FASE 3: OPTIMIZACIÓN SAC
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 3: OPTIMIZACIÓN SAC (75 trials)")
    logger.info("=" * 60)
    
    best_params, study = train_sac_optuna(train_data, val_data, n_trials=75, device=device)
    
    logger.info(f"\n🎯 Mejores parámetros: {best_params}")
    
    # =========================================
    # FASE 4: ENTRENAR MODELO FINAL
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 4: ENTRENANDO MODELO FINAL")
    logger.info("=" * 60)
    
    # Combinar train + val
    combined_data = {
        'embeddings': np.vstack([train_data['embeddings'], val_data['embeddings']]),
        'pred_4h': np.vstack([train_data['pred_4h'], val_data['pred_4h']]),
        'pred_12h': np.vstack([train_data['pred_12h'], val_data['pred_12h']]),
        'pred_24h': np.vstack([train_data['pred_24h'], val_data['pred_24h']]),
        'regimes': np.concatenate([train_data['regimes'], val_data['regimes']]),
        'prices': np.concatenate([train_data['prices'], val_data['prices']]),
        'atrs': np.concatenate([train_data['atrs'], val_data['atrs']])
    }
    
    final_config = AdvancedTradingConfig(
        max_leverage=best_params['max_leverage'],
        kelly_fraction=best_params['kelly_fraction'],
        atr_multiplier_stop=best_params['atr_stop'],
        atr_multiplier_trail=best_params['atr_trail'],
        max_drawdown=best_params['max_drawdown'],
        min_position=best_params['min_position'],
        max_position=best_params['max_position'],
    )
    
    final_env = AdvancedTradingEnv(
        combined_data['embeddings'], combined_data['pred_4h'], combined_data['pred_12h'],
        combined_data['pred_24h'], combined_data['regimes'], combined_data['prices'],
        combined_data['atrs'], final_config
    )
    
    final_agent = SACAdvanced(
        state_dim=final_env.state_dim,
        hidden_dim=best_params['hidden_dim'],
        lr=best_params['lr'],
        gamma=best_params['gamma'],
        tau=best_params['tau'],
        device=device
    )
    
    final_agent.train(
        env=final_env,
        total_steps=200000,
        batch_size=best_params['batch_size'],
        start_steps=10000,
        log_interval=50000
    )
    
    # Guardar
    (root_dir / "artifacts/final_pro").mkdir(parents=True, exist_ok=True)
    final_agent.save(str(root_dir / "artifacts/final_pro/sac_agent.pt"))
    
    # =========================================
    # FASE 5: EVALUACIÓN EN HOLDOUT
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   📊 EVALUACIÓN FINAL EN HOLDOUT")
    logger.info("=" * 60)
    
    holdout_env = AdvancedTradingEnv(
        holdout_data['embeddings'], holdout_data['pred_4h'], holdout_data['pred_12h'],
        holdout_data['pred_24h'], holdout_data['regimes'], holdout_data['prices'],
        holdout_data['atrs'], final_config
    )
    
    holdout_metrics = final_agent.evaluate(holdout_env, n_episodes=20)
    
    # =========================================
    # RESUMEN FINAL
    # =========================================
    print("\n" + "=" * 70)
    print("              ✅ ENTRENAMIENTO PRO COMPLETADO")
    print("=" * 70)
    
    print(f"\n📊 RESULTADOS EN HOLDOUT:")
    print(f"   Retorno medio: {holdout_metrics['mean_return']*100:+.2f}%")
    print(f"   Sharpe ratio: {holdout_metrics['sharpe']:.2f}")
    print(f"   Win rate: {holdout_metrics['mean_win_rate']*100:.1f}%")
    print(f"   Max drawdown: {holdout_metrics['max_drawdown']*100:.1f}%")
    print(f"   Trades promedio: {holdout_metrics['mean_trades']:.0f}")
    
    print(f"\n🎯 Mejores hiperparámetros:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    print(f"\n💾 Artefactos guardados en artifacts/final_pro/")
    
    # Guardar métricas
    np.savez(
        root_dir / "artifacts/final_pro/holdout_metrics.npz",
        mean_return=holdout_metrics['mean_return'],
        sharpe=holdout_metrics['sharpe'],
        win_rate=holdout_metrics['mean_win_rate'],
        max_drawdown=holdout_metrics['max_drawdown'],
        trades=holdout_metrics['mean_trades']
    )
    
    # Evaluación
    if holdout_metrics['sharpe'] > 1.0 and holdout_metrics['mean_return'] > 0:
        print("\n🎉 ¡MODELO EXITOSO! Listo para paper trading.")
    elif holdout_metrics['sharpe'] > 0 and holdout_metrics['mean_return'] > -0.02:
        print("\n🟡 Modelo aceptable. Considerar más optimización.")
    else:
        print("\n🔴 Modelo necesita mejoras.")


if __name__ == "__main__":
    main()
