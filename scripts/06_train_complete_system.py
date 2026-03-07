"""
Script 06: Entrenamiento del Sistema Completo.

Fases:
1. Pre-entrenamiento Contrastivo
2. Inicialización Chronos
3. Entrenamiento Fusión Multi-Timeframe
4. Entrenamiento SAC Advanced
5. Optimización con Optuna
6. Evaluación en Holdout

Uso:
    python scripts/06_train_complete_system.py
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
from datetime import datetime
from typing import Dict, List, Tuple

from src.features.indicators import calculate_all_indicators
from src.models.contrastive_pretrain import (
    ContrastiveEncoder, ContrastiveConfig, 
    ContrastiveDataset, ContrastiveTrainer
)
from src.models.complete_system import (
    CompleteSystem, CompleteSystemConfig,
    FusionNetwork, ChronosPredictor, CHRONOS_AVAILABLE
)
from src.models.sac_advanced import (
    SACAdvanced, AdvancedTradingEnv, AdvancedTradingConfig,
    KellyCriterion, TrailingStopManager
)


class MultiTimeframeDatasetComplete(Dataset):
    """Dataset completo con las 3 temporalidades + Chronos features."""
    
    def __init__(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        chronos_predictor: ChronosPredictor = None,
        seq_len_1h: int = 24,
        seq_len_4h: int = 50,
        seq_len_1d: int = 30
    ):
        self.seq_len_1h = seq_len_1h
        self.seq_len_4h = seq_len_4h
        self.seq_len_1d = seq_len_1d
        self.chronos = chronos_predictor
        
        # Feature columns
        self.feature_cols = [
            'return_1', 'return_4', 'return_24',
            'rsi', 'macd_hist', 'bb_position', 'atr_pct',
            'adx', 'di_plus', 'di_minus',
            'vol_ratio', 'trend_ema', 'trend_long', 'trend_strength'
        ]
        
        # Preparar datos
        self.data_1h = self._prepare(df_1h)
        self.data_4h = self._prepare(df_4h)
        self.data_1d = self._prepare(df_1d)
        
        # Timestamps y precios
        self.ts_1h = pd.to_datetime(df_1h['timestamp']).values
        self.ts_4h = pd.to_datetime(df_4h['timestamp']).values
        self.ts_1d = pd.to_datetime(df_1d['timestamp']).values
        
        self.prices_4h = df_4h['close'].values
        self.prices_1h = df_1h['close'].values
        
        # ATR para trailing stop
        if 'atr' in df_4h.columns:
            self.atrs = df_4h['atr'].values
        else:
            self.atrs = self._calc_atr(df_4h)
        
        # Retornos futuros (targets)
        self.returns_4h = self._future_returns(self.prices_4h, 1)
        self.returns_12h = self._future_returns(self.prices_4h, 3)
        self.returns_24h = self._future_returns(self.prices_4h, 6)
        
        # Régimen de mercado
        self.regimes = self._calc_regimes(df_4h)
        
        # Índices válidos
        self._calc_valid_indices()
        
        # Pre-calcular Chronos features (costoso, hacerlo una vez)
        self.chronos_features = self._precompute_chronos()
        
        logger.info(f"📊 Dataset completo creado: {len(self.valid_indices)} samples")
    
    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.feature_cols if c in df.columns]
        data = df[cols].values.astype(np.float32)
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0) + 1e-8
        data = (data - mean) / std
        return np.nan_to_num(data, 0)
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr = np.maximum(high[1:] - low[1:], 
                       np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
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
                regimes[i] = 3  # High volatility
            elif returns[i] > 0.02 and adx[i] > 25:
                regimes[i] = 0  # Trend up
            elif returns[i] < -0.02 and adx[i] > 25:
                regimes[i] = 1  # Trend down
            else:
                regimes[i] = 2  # Ranging
        
        return regimes
    
    def _calc_valid_indices(self):
        self.valid_indices = []
        self.alignment = []
        
        min_1h = self.seq_len_1h
        min_4h = self.seq_len_4h
        min_1d = self.seq_len_1d
        
        for idx_4h in range(min_4h, len(self.data_4h) - 6):
            ts = self.ts_4h[idx_4h]
            
            # Encontrar índices correspondientes
            mask_1d = self.ts_1d <= ts
            if mask_1d.sum() < min_1d:
                continue
            idx_1d = mask_1d.sum() - 1
            
            mask_1h = self.ts_1h <= ts
            if mask_1h.sum() < min_1h:
                continue
            idx_1h = mask_1h.sum() - 1
            
            self.valid_indices.append(idx_4h)
            self.alignment.append((idx_1d, idx_1h))
    
    def _precompute_chronos(self) -> np.ndarray:
        """Pre-computa Chronos features para todos los índices válidos."""
        if self.chronos is None or not self.chronos.available:
            return np.zeros((len(self.valid_indices), 36), dtype=np.float32)
        
        logger.info("🔮 Pre-computando Chronos features...")
        features = []
        
        for i, idx_4h in enumerate(self.valid_indices):
            if i % 1000 == 0:
                logger.info(f"   Progreso: {i}/{len(self.valid_indices)}")
            
            prices = self.prices_4h[:idx_4h + 1]
            preds = self.chronos.predict(prices[-200:], prediction_length=12)
            
            feat = np.concatenate([preds['q10'], preds['q50'], preds['q90']])
            features.append(feat)
        
        logger.info("   ✅ Chronos features pre-computados")
        return np.array(features, dtype=np.float32)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        idx_4h = self.valid_indices[idx]
        idx_1d, idx_1h = self.alignment[idx]
        
        # Secuencias
        x_1h = self.data_1h[idx_1h - self.seq_len_1h + 1:idx_1h + 1]
        x_4h = self.data_4h[idx_4h - self.seq_len_4h + 1:idx_4h + 1]
        x_1d = self.data_1d[idx_1d - self.seq_len_1d + 1:idx_1d + 1]
        
        # Chronos
        chronos_feat = self.chronos_features[idx]
        
        # Targets
        returns = np.array([
            [self.returns_4h[idx_4h], self.returns_12h[idx_4h], self.returns_24h[idx_4h]]
        ], dtype=np.float32).flatten()
        
        return {
            'x_1h': torch.tensor(x_1h, dtype=torch.float32),
            'x_4h': torch.tensor(x_4h, dtype=torch.float32),
            'x_1d': torch.tensor(x_1d, dtype=torch.float32),
            'chronos': torch.tensor(chronos_feat, dtype=torch.float32),
            'returns': torch.tensor(returns, dtype=torch.float32),
            'regime': torch.tensor(self.regimes[idx_4h], dtype=torch.long),
            'price': torch.tensor(self.prices_4h[idx_4h], dtype=torch.float32),
            'atr': torch.tensor(self.atrs[idx_4h], dtype=torch.float32)
        }


class FusionTrainer:
    """Entrena la red de fusión."""
    
    def __init__(self, model: FusionNetwork, lr: float = 1e-3, device: str = 'cuda'):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pinball loss para cuantiles."""
        # pred: (batch, 3, 3) - 3 horizontes, 3 cuantiles
        # target: (batch, 3) - 3 horizontes
        quantiles = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        loss = 0
        for h in range(3):  # horizontes
            for q_idx, q in enumerate(quantiles):
                error = target[:, h] - pred[:, h, q_idx]
                loss += torch.max(q * error, (q - 1) * error).mean()
        return loss / 9
    
    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        regime_correct = 0
        total = 0
        
        for batch in loader:
            x_1h = batch['x_1h'].to(self.device)
            x_4h = batch['x_4h'].to(self.device)
            x_1d = batch['x_1d'].to(self.device)
            chronos = batch['chronos'].to(self.device)
            returns = batch['returns'].to(self.device)
            regime = batch['regime'].to(self.device)
            
            # Forward
            out = self.model(x_1h, x_4h, x_1d, chronos)
            
            # Losses
            loss_ret = self.quantile_loss(out['return_preds'], returns.view(-1, 3))
            loss_regime = torch.nn.functional.cross_entropy(out['regime_logits'], regime)
            loss = loss_ret + 0.3 * loss_regime
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            regime_correct += (out['regime_logits'].argmax(dim=-1) == regime).sum().item()
            total += len(regime)
        
        return {
            'loss': total_loss / len(loader),
            'regime_acc': regime_correct / total
        }
    
    def validate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        regime_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                x_1h = batch['x_1h'].to(self.device)
                x_4h = batch['x_4h'].to(self.device)
                x_1d = batch['x_1d'].to(self.device)
                chronos = batch['chronos'].to(self.device)
                returns = batch['returns'].to(self.device)
                regime = batch['regime'].to(self.device)
                
                out = self.model(x_1h, x_4h, x_1d, chronos)
                
                loss_ret = self.quantile_loss(out['return_preds'], returns.view(-1, 3))
                total_loss += loss_ret.item()
                regime_correct += (out['regime_logits'].argmax(dim=-1) == regime).sum().item()
                total += len(regime)
        
        self.scheduler.step(total_loss / len(loader))
        
        return {
            'val_loss': total_loss / len(loader),
            'regime_acc': regime_correct / total
        }
    
    def train(self, train_loader, val_loader, epochs: int = 100) -> Dict:
        logger.info(f"🏋️ Entrenando Fusión: {epochs} épocas")
        
        best_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_m = self.train_epoch(train_loader)
            val_m = self.validate(val_loader)
            
            history['train_loss'].append(train_m['loss'])
            history['val_loss'].append(val_m['val_loss'])
            
            if val_m['val_loss'] < best_loss:
                best_loss = val_m['val_loss']
                self.best_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Época {epoch+1}/{epochs} | "
                    f"Train: {train_m['loss']:.4f} | "
                    f"Val: {val_m['val_loss']:.4f} | "
                    f"Regime Acc: {val_m['regime_acc']:.1%}"
                )
        
        self.model.load_state_dict(self.best_state)
        logger.info(f"✅ Fusión entrenada. Best val loss: {best_loss:.4f}")
        
        return history


def generate_embeddings(
    model: FusionNetwork,
    dataset: MultiTimeframeDatasetComplete,
    device: str
) -> Dict[str, np.ndarray]:
    """Genera embeddings y predicciones para SAC."""
    model.eval()
    
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    embeddings = []
    pred_4h = []
    pred_12h = []
    pred_24h = []
    regimes = []
    prices = []
    atrs = []
    
    with torch.no_grad():
        for batch in loader:
            x_1h = batch['x_1h'].to(device)
            x_4h = batch['x_4h'].to(device)
            x_1d = batch['x_1d'].to(device)
            chronos = batch['chronos'].to(device)
            
            out = model(x_1h, x_4h, x_1d, chronos)
            
            embeddings.append(out['embedding'].cpu().numpy())
            
            # Predicciones por horizonte
            preds = out['return_preds'].cpu().numpy()
            pred_4h.append(preds[:, 0, :])   # Horizonte 0
            pred_12h.append(preds[:, 1, :])  # Horizonte 1
            pred_24h.append(preds[:, 2, :])  # Horizonte 2
            
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


def train_sac_with_optuna(
    train_data: Dict,
    val_data: Dict,
    n_trials: int = 50,
    device: str = 'cuda'
) -> Tuple[SACAdvanced, Dict]:
    """Entrena SAC con optimización de hiperparámetros."""
    
    logger.info(f"🔧 Optimización Optuna: {n_trials} trials")
    
    def objective(trial):
        # Hiperparámetros de trading
        config = AdvancedTradingConfig(
            max_leverage=trial.suggest_int('max_leverage', 2, 5),
            kelly_fraction=trial.suggest_float('kelly_fraction', 0.3, 1.0),
            atr_multiplier_stop=trial.suggest_float('atr_stop', 1.5, 3.5),
            atr_multiplier_trail=trial.suggest_float('atr_trail', 1.0, 2.5),
            max_drawdown=trial.suggest_float('max_drawdown', 0.10, 0.25),
            min_position=trial.suggest_float('min_position', 0.05, 0.2),
            max_position=trial.suggest_float('max_position', 0.7, 1.0),
        )
        
        # Hiperparámetros de red
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        tau = trial.suggest_float('tau', 0.001, 0.01)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        
        # Crear entorno de validación
        env = AdvancedTradingEnv(
            val_data['embeddings'],
            val_data['pred_4h'],
            val_data['pred_12h'],
            val_data['pred_24h'],
            val_data['regimes'],
            val_data['prices'],
            val_data['atrs'],
            config
        )
        
        # Crear agente
        agent = SACAdvanced(
            state_dim=env.state_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            tau=tau,
            device=device
        )
        
        # Entrenar
        try:
            agent.train(
                env=env,
                total_steps=40000,
                batch_size=batch_size,
                start_steps=2000,
                log_interval=100000
            )
            
            # Evaluar
            metrics = agent.evaluate(env, n_episodes=5)
            
            score = metrics['sharpe']
            
            # Penalizaciones
            if metrics['mean_return'] < 0:
                score -= 2.0
            if metrics['max_drawdown'] > 0.15:
                score -= 1.0
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -10.0
    
    # Optimizar
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"✅ Mejor Sharpe: {study.best_value:.2f}")
    logger.info(f"🎯 Mejores parámetros: {study.best_params}")
    
    return study.best_params, study


def train_final_model(
    train_data: Dict,
    best_params: Dict,
    total_steps: int = 300000,
    device: str = 'cuda'
) -> SACAdvanced:
    """Entrena el modelo final con los mejores parámetros."""
    
    logger.info("🚀 Entrenando modelo final...")
    
    config = AdvancedTradingConfig(
        max_leverage=best_params['max_leverage'],
        kelly_fraction=best_params['kelly_fraction'],
        atr_multiplier_stop=best_params['atr_stop'],
        atr_multiplier_trail=best_params['atr_trail'],
        max_drawdown=best_params['max_drawdown'],
        min_position=best_params['min_position'],
        max_position=best_params['max_position'],
    )
    
    env = AdvancedTradingEnv(
        train_data['embeddings'],
        train_data['pred_4h'],
        train_data['pred_12h'],
        train_data['pred_24h'],
        train_data['regimes'],
        train_data['prices'],
        train_data['atrs'],
        config
    )
    
    agent = SACAdvanced(
        state_dim=env.state_dim,
        hidden_dim=best_params['hidden_dim'],
        lr=best_params['lr'],
        gamma=best_params['gamma'],
        tau=best_params['tau'],
        device=device
    )
    
    agent.train(
        env=env,
        total_steps=total_steps,
        batch_size=best_params['batch_size'],
        start_steps=10000,
        log_interval=50000
    )
    
    return agent, config


def evaluate_on_holdout(
    agent: SACAdvanced,
    holdout_data: Dict,
    config: AdvancedTradingConfig
) -> Dict:
    """Evaluación final en holdout (nunca visto)."""
    
    logger.info("\n" + "=" * 60)
    logger.info("   📊 EVALUACIÓN FINAL EN HOLDOUT")
    logger.info("=" * 60)
    
    env = AdvancedTradingEnv(
        holdout_data['embeddings'],
        holdout_data['pred_4h'],
        holdout_data['pred_12h'],
        holdout_data['pred_24h'],
        holdout_data['regimes'],
        holdout_data['prices'],
        holdout_data['atrs'],
        config
    )
    
    metrics = agent.evaluate(env, n_episodes=20)
    
    return metrics


def main():
    print("\n" + "=" * 70)
    print("       🚀 ENTRENAMIENTO SISTEMA COMPLETO - MÁXIMA OPTIMIZACIÓN")
    print("=" * 70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Device: {device}")
    
    # =========================================
    # CARGAR DATOS
    # =========================================
    logger.info("\n📥 Cargando datos...")
    
    df_1h = pd.read_csv(root_dir / "data/raw/btcusdt_1h.csv")
    df_4h = pd.read_csv(root_dir / "data/raw/btcusdt_4h.csv")
    df_1d = pd.read_csv(root_dir / "data/raw/btcusdt_1d.csv")
    
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
    
    logger.info(f"   1H: {len(df_1h):,} velas ({df_1h['timestamp'].min()} - {df_1h['timestamp'].max()})")
    logger.info(f"   4H: {len(df_4h):,} velas")
    logger.info(f"   1D: {len(df_1d):,} velas")
    
    # Indicadores
    logger.info("📊 Calculando indicadores...")
    df_1h = calculate_all_indicators(df_1h)
    df_4h = calculate_all_indicators(df_4h)
    df_1d = calculate_all_indicators(df_1d)
    
    # =========================================
    # FASE 1: PRE-ENTRENAMIENTO CONTRASTIVO
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 1: PRE-ENTRENAMIENTO CONTRASTIVO")
    logger.info("=" * 60)
    
    contra_dataset = ContrastiveDataset(df_4h, seq_len=50)
    contra_loader = DataLoader(
        contra_dataset, batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    sample = contra_dataset[0]['anchor']
    input_dim = sample.shape[-1]
    
    contra_encoder = ContrastiveEncoder(
        input_dim=input_dim,
        hidden_dim=256,
        embedding_dim=128
    )
    
    contra_config = ContrastiveConfig(embedding_dim=128, temperature=0.07)
    contra_trainer = ContrastiveTrainer(contra_encoder, contra_config, device)
    contra_trainer.train(contra_loader, epochs=50)
    
    # Guardar
    (root_dir / "artifacts/contrastive").mkdir(parents=True, exist_ok=True)
    contra_trainer.save(str(root_dir / "artifacts/contrastive/encoder.pt"))
    
    # =========================================
    # FASE 2: INICIALIZAR CHRONOS
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 2: INICIALIZANDO CHRONOS")
    logger.info("=" * 60)
    
    chronos = ChronosPredictor(device=device) if CHRONOS_AVAILABLE else None
    
    # =========================================
    # FASE 3: CREAR DATASET COMPLETO
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 3: CREANDO DATASET COMPLETO")
    logger.info("=" * 60)
    
    full_dataset = MultiTimeframeDatasetComplete(
        df_1h, df_4h, df_1d,
        chronos_predictor=chronos
    )
    
    # Split: 50% train, 20% val, 15% test, 15% holdout
    n = len(full_dataset)
    train_end = int(n * 0.50)
    val_end = int(n * 0.70)
    test_end = int(n * 0.85)
    
    train_indices = list(range(train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, test_end))
    holdout_indices = list(range(test_end, n))
    
    logger.info(f"   Train: {len(train_indices)}")
    logger.info(f"   Val: {len(val_indices)}")
    logger.info(f"   Test: {len(test_indices)}")
    logger.info(f"   Holdout: {len(holdout_indices)} (INTOCABLE)")
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # =========================================
    # FASE 4: ENTRENAR RED DE FUSIÓN
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 4: ENTRENANDO RED DE FUSIÓN")
    logger.info("=" * 60)
    
    fusion_model = FusionNetwork(
        input_dim_1h=input_dim,
        input_dim_4h=input_dim,
        input_dim_1d=input_dim,
        chronos_dim=36,
        hidden_dim=256,
        output_dim=128
    )
    
    fusion_trainer = FusionTrainer(fusion_model, lr=1e-3, device=device)
    fusion_trainer.train(train_loader, val_loader, epochs=100)
    
    # Guardar
    (root_dir / "artifacts/fusion").mkdir(parents=True, exist_ok=True)
    torch.save(fusion_model.state_dict(), root_dir / "artifacts/fusion/model.pt")
    
    # =========================================
    # FASE 5: GENERAR EMBEDDINGS
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 5: GENERANDO EMBEDDINGS")
    logger.info("=" * 60)
    
    # Subset datasets para generar embeddings
    train_sub = torch.utils.data.Subset(full_dataset, train_indices)
    val_sub = torch.utils.data.Subset(full_dataset, val_indices)
    test_sub = torch.utils.data.Subset(full_dataset, test_indices)
    holdout_sub = torch.utils.data.Subset(full_dataset, holdout_indices)
    
    train_data = generate_embeddings(fusion_model, train_sub, device)
    val_data = generate_embeddings(fusion_model, val_sub, device)
    test_data = generate_embeddings(fusion_model, test_sub, device)
    holdout_data = generate_embeddings(fusion_model, holdout_sub, device)
    
    logger.info(f"   Train embeddings: {train_data['embeddings'].shape}")
    logger.info(f"   Val embeddings: {val_data['embeddings'].shape}")
    logger.info(f"   Holdout embeddings: {holdout_data['embeddings'].shape}")
    
    # =========================================
    # FASE 6: OPTIMIZACIÓN SAC CON OPTUNA
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 6: OPTIMIZACIÓN SAC CON OPTUNA")
    logger.info("=" * 60)
    
    best_params, study = train_sac_with_optuna(
        train_data, val_data,
        n_trials=50,
        device=device
    )
    
    # Guardar estudio
    import pickle
    with open(root_dir / "artifacts/optuna_study_complete.pkl", 'wb') as f:
        pickle.dump(study, f)
    
    # =========================================
    # FASE 7: ENTRENAR MODELO FINAL
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 7: ENTRENANDO MODELO FINAL")
    logger.info("=" * 60)
    
    # Combinar train + val para entrenamiento final
    combined_data = {
        'embeddings': np.vstack([train_data['embeddings'], val_data['embeddings']]),
        'pred_4h': np.vstack([train_data['pred_4h'], val_data['pred_4h']]),
        'pred_12h': np.vstack([train_data['pred_12h'], val_data['pred_12h']]),
        'pred_24h': np.vstack([train_data['pred_24h'], val_data['pred_24h']]),
        'regimes': np.concatenate([train_data['regimes'], val_data['regimes']]),
        'prices': np.concatenate([train_data['prices'], val_data['prices']]),
        'atrs': np.concatenate([train_data['atrs'], val_data['atrs']])
    }
    
    final_agent, final_config = train_final_model(
        combined_data, best_params,
        total_steps=300000,
        device=device
    )
    
    # Guardar
    (root_dir / "artifacts/final").mkdir(parents=True, exist_ok=True)
    final_agent.save(str(root_dir / "artifacts/final/sac_agent.pt"))
    
    # =========================================
    # FASE 8: EVALUACIÓN FINAL EN HOLDOUT
    # =========================================
    holdout_metrics = evaluate_on_holdout(final_agent, holdout_data, final_config)
    
    # =========================================
    # RESUMEN FINAL
    # =========================================
    print("\n" + "=" * 70)
    print("              ✅ ENTRENAMIENTO COMPLETO FINALIZADO")
    print("=" * 70)
    
    print(f"\n📊 RESULTADOS EN HOLDOUT (datos nunca vistos):")
    print(f"   Retorno medio: {holdout_metrics['mean_return']*100:+.2f}%")
    print(f"   Sharpe ratio: {holdout_metrics['sharpe']:.2f}")
    print(f"   Win rate: {holdout_metrics['mean_win_rate']*100:.1f}%")
    print(f"   Max drawdown: {holdout_metrics['max_drawdown']*100:.1f}%")
    print(f"   Trades promedio: {holdout_metrics['mean_trades']:.0f}")
    
    print(f"\n🎯 Mejores hiperparámetros:")
    for k, v in best_params.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    
    print(f"\n💾 Artefactos guardados:")
    print(f"   artifacts/contrastive/encoder.pt")
    print(f"   artifacts/fusion/model.pt")
    print(f"   artifacts/final/sac_agent.pt")
    print(f"   artifacts/optuna_study_complete.pkl")
    
    # Guardar métricas
    np.savez(
        root_dir / "artifacts/final/metrics.npz",
        **holdout_metrics,
        **best_params
    )
    
    print(f"\n🎉 Sistema listo para backtest final:")
    print(f"   python scripts/07_final_backtest.py")


if __name__ == "__main__":
    main()
