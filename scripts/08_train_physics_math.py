"""
Script 08: Entrenamiento con Física + Matemáticas Avanzadas.

147 features totales:
- 14 básicas
- 35 PRO (VWAP, Fibonacci, Market Structure)
- 47 física (momentum, energía, fractales)
- 51 matemáticas (wavelets, información mutua, copulas)

Este es el sistema más avanzado posible.
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
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from src.features.indicators import calculate_all_indicators
from src.features.indicators_pro import calculate_pro_indicators
from src.features.physics_features import calculate_physics_features
from src.features.advanced_math import calculate_advanced_math_features
from src.models.sac_advanced import SACAdvanced, AdvancedTradingEnv, AdvancedTradingConfig


class UltimateDataset(Dataset):
    """Dataset con TODAS las features (147 total)."""
    
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
        
        # Preparar datos con TODAS las features
        logger.info("📊 Preparando datos con 147 features...")
        self.data_1h, self.cols_1h = self._prepare_all_features(df_1h, "1H")
        self.data_4h, self.cols_4h = self._prepare_all_features(df_4h, "4H")
        self.data_1d, self.cols_1d = self._prepare_all_features(df_1d, "1D")
        
        # Timestamps y precios
        self.ts_1h = pd.to_datetime(df_1h['timestamp']).values
        self.ts_4h = pd.to_datetime(df_4h['timestamp']).values
        self.ts_1d = pd.to_datetime(df_1d['timestamp']).values
        
        self.prices_4h = df_4h['close'].values
        self.atrs = df_4h['atr'].values if 'atr' in df_4h.columns else self._calc_atr(df_4h)
        
        # Targets
        self.returns_4h = self._future_returns(self.prices_4h, 1)
        self.returns_12h = self._future_returns(self.prices_4h, 3)
        self.returns_24h = self._future_returns(self.prices_4h, 6)
        
        # Régimen
        self.regimes = self._calc_regimes(df_4h)
        
        # Índices válidos
        self._calc_valid_indices()
        
        logger.info(f"✅ UltimateDataset: {len(self.valid_indices)} samples")
        logger.info(f"   Features 1H: {self.data_1h.shape[1]}")
        logger.info(f"   Features 4H: {self.data_4h.shape[1]}")
        logger.info(f"   Features 1D: {self.data_1d.shape[1]}")
    
    def _prepare_all_features(self, df: pd.DataFrame, name: str) -> tuple:
        """Calcula TODAS las features para un timeframe."""
        logger.info(f"   Calculando features {name}...")
        
        # Excluir columnas base
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
        
        # Obtener columnas de features
        feature_cols = [c for c in df.columns if c not in exclude]
        
        if not feature_cols:
            logger.warning(f"   ⚠️ No hay features en {name}, usando solo OHLCV")
            feature_cols = ['close', 'volume']
        
        data = df[feature_cols].values.astype(np.float32)
        
        # Normalización robusta
        median = np.nanmedian(data, axis=0)
        mad = np.nanmedian(np.abs(data - median), axis=0) + 1e-8
        data = (data - median) / (mad * 1.4826)
        
        # Clip outliers
        data = np.clip(data, -5, 5)
        data = np.nan_to_num(data, 0)
        
        logger.info(f"   {name}: {len(feature_cols)} features")
        
        return data, feature_cols
    
    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        tr = np.insert(tr, 0, high[0] - low[0])
        return pd.Series(tr).rolling(period).mean().fillna(method='bfill').values
    
    def _future_returns(self, prices: np.ndarray, horizon: int) -> np.ndarray:
        returns = np.zeros(len(prices))
        for i in range(len(prices) - horizon):
            returns[i] = (prices[i + horizon] - prices[i]) / prices[i]
        return returns.astype(np.float32)
    
    def _calc_regimes(self, df: pd.DataFrame) -> np.ndarray:
        regimes = np.zeros(len(df), dtype=np.int64)
        returns = df['close'].pct_change(20).values
        volatility = df['close'].pct_change().rolling(20).std().values
        
        # Usar Hurst si está disponible
        if 'hurst' in df.columns:
            hurst = df['hurst'].values
        else:
            hurst = np.full(len(df), 0.5)
        
        vol_threshold = np.nanpercentile(volatility, 75)
        
        for i in range(len(df)):
            if np.isnan(returns[i]) or np.isnan(volatility[i]):
                regimes[i] = 2
                continue
            
            if volatility[i] > vol_threshold:
                regimes[i] = 3  # High volatility
            elif hurst[i] > 0.6 and returns[i] > 0.01:
                regimes[i] = 0  # Trending up
            elif hurst[i] > 0.6 and returns[i] < -0.01:
                regimes[i] = 1  # Trending down
            else:
                regimes[i] = 2  # Ranging/mean reverting
        
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
    def input_dim_1h(self):
        return self.data_1h.shape[1]
    
    @property
    def input_dim_4h(self):
        return self.data_4h.shape[1]
    
    @property
    def input_dim_1d(self):
        return self.data_1d.shape[1]


class UltimateFusionNetwork(torch.nn.Module):
    """Red de fusión para 147 features."""
    
    def __init__(self, input_dim_1h: int, input_dim_4h: int, input_dim_1d: int,
                 hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        
        # Feature reduction (muchas features → representación compacta)
        self.reduce_1h = torch.nn.Sequential(
            torch.nn.Linear(input_dim_1h, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.reduce_4h = torch.nn.Sequential(
            torch.nn.Linear(input_dim_4h, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.reduce_1d = torch.nn.Sequential(
            torch.nn.Linear(input_dim_1d, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # LSTMs
        self.lstm_1h = torch.nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2,
                                      batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm_4h = torch.nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2,
                                      batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm_1d = torch.nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2,
                                      batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention
        self.attention = torch.nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        # Fusion
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim * 2),
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Heads
        self.regime_head = torch.nn.Linear(hidden_dim, 4)
        self.return_head = torch.nn.Linear(hidden_dim, 9)
        self.embedding_head = torch.nn.Linear(hidden_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, x_1h, x_4h, x_1d):
        # Reduce dimensionality
        x_1h = self.reduce_1h(x_1h)
        x_4h = self.reduce_4h(x_4h)
        x_1d = self.reduce_1d(x_1d)
        
        # LSTM
        _, (h_1h, _) = self.lstm_1h(x_1h)
        _, (h_4h, _) = self.lstm_4h(x_4h)
        _, (h_1d, _) = self.lstm_1d(x_1d)
        
        emb_1h = torch.cat([h_1h[-2], h_1h[-1]], dim=-1)
        emb_4h = torch.cat([h_4h[-2], h_4h[-1]], dim=-1)
        emb_1d = torch.cat([h_1d[-2], h_1d[-1]], dim=-1)
        
        # Stack for attention
        stacked = torch.stack([emb_1h, emb_4h, emb_1d], dim=1)
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Flatten
        combined = attended.reshape(attended.size(0), -1)
        fused = self.fusion(combined)
        
        return {
            'embedding': self.embedding_head(fused),
            'regime_logits': self.regime_head(fused),
            'return_preds': self.return_head(fused).view(-1, 3, 3)
        }


def train_fusion_network(model, train_loader, val_loader, epochs=100, device='cuda'):
    """Entrena la red de fusión."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    
    best_loss = float('inf')
    patience = 0
    max_patience = 25
    
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
            
            loss_ret = torch.nn.functional.huber_loss(out['return_preds'][:, :, 1], returns)
            loss_regime = torch.nn.functional.cross_entropy(out['regime_logits'], regime)
            loss = loss_ret + 0.2 * loss_regime
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
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
                loss = torch.nn.functional.huber_loss(out['return_preds'][:, :, 1], returns)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Época {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if patience >= max_patience:
            logger.info(f"Early stopping en época {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model


def generate_embeddings(model, dataset, device):
    """Genera embeddings."""
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


def main():
    print("\n" + "=" * 70)
    print("    🚀 ENTRENAMIENTO ULTIMATE: FÍSICA + MATEMÁTICAS (147 FEATURES)")
    print("=" * 70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️ Device: {device}")
    
    # =========================================
    # CARGAR Y PROCESAR DATOS
    # =========================================
    logger.info("\n📥 Cargando datos...")
    
    df_1h = pd.read_csv(root_dir / "data/raw/btcusdt_1h.csv")
    df_4h = pd.read_csv(root_dir / "data/raw/btcusdt_4h.csv")
    df_1d = pd.read_csv(root_dir / "data/raw/btcusdt_1d.csv")
    
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
    
    # Calcular TODAS las features
    logger.info("\n📊 Calculando TODAS las features...")
    
    for name, df in [("1H", df_1h), ("4H", df_4h), ("1D", df_1d)]:
        logger.info(f"\n   {name}:")
        df_temp = calculate_all_indicators(df.copy())
        df_temp = calculate_pro_indicators(df_temp)
        df_temp = calculate_physics_features(df_temp)
        df_temp = calculate_advanced_math_features(df_temp)
        
        if name == "1H":
            df_1h = df_temp
        elif name == "4H":
            df_4h = df_temp
        else:
            df_1d = df_temp
    
    # =========================================
    # CREAR DATASET
    # =========================================
    logger.info("\n📦 Creando UltimateDataset...")
    
    full_dataset = UltimateDataset(df_1h, df_4h, df_1d)
    
    # Split
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
    logger.info(f"   Holdout: {len(holdout_idx)}")
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # =========================================
    # ENTRENAR RED DE FUSIÓN
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 1: ENTRENANDO RED DE FUSIÓN ULTIMATE")
    logger.info("=" * 60)
    
    fusion_model = UltimateFusionNetwork(
        input_dim_1h=full_dataset.input_dim_1h,
        input_dim_4h=full_dataset.input_dim_4h,
        input_dim_1d=full_dataset.input_dim_1d,
        hidden_dim=256,
        output_dim=128,
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in fusion_model.parameters())
    logger.info(f"   Parámetros totales: {total_params:,}")
    
    fusion_model = train_fusion_network(fusion_model, train_loader, val_loader, epochs=100, device=device)
    
    # Guardar
    (root_dir / "artifacts/ultimate").mkdir(parents=True, exist_ok=True)
    torch.save(fusion_model.state_dict(), root_dir / "artifacts/ultimate/fusion_model.pt")
    
    # =========================================
    # GENERAR EMBEDDINGS
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 2: GENERANDO EMBEDDINGS")
    logger.info("=" * 60)
    
    train_data = generate_embeddings(fusion_model, torch.utils.data.Subset(full_dataset, train_idx), device)
    val_data = generate_embeddings(fusion_model, torch.utils.data.Subset(full_dataset, val_idx), device)
    holdout_data = generate_embeddings(fusion_model, torch.utils.data.Subset(full_dataset, holdout_idx), device)
    
    # =========================================
    # OPTIMIZACIÓN SAC
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 3: OPTIMIZACIÓN SAC (100 trials)")
    logger.info("=" * 60)
    
    def objective(trial):
        config = AdvancedTradingConfig(
            max_leverage=trial.suggest_int('max_leverage', 2, 3),
            kelly_fraction=trial.suggest_float('kelly_fraction', 0.15, 0.4),
            atr_multiplier_stop=trial.suggest_float('atr_stop', 2.0, 3.5),
            atr_multiplier_trail=trial.suggest_float('atr_trail', 1.5, 2.5),
            max_drawdown=trial.suggest_float('max_drawdown', 0.06, 0.12),
            min_position=trial.suggest_float('min_position', 0.03, 0.10),
            max_position=trial.suggest_float('max_position', 0.4, 0.7),
        )
        
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256])
        lr = trial.suggest_float('lr', 3e-5, 3e-4, log=True)
        gamma = trial.suggest_float('gamma', 0.98, 0.999)
        tau = trial.suggest_float('tau', 0.003, 0.01)
        
        env = AdvancedTradingEnv(
            val_data['embeddings'], val_data['pred_4h'], val_data['pred_12h'],
            val_data['pred_24h'], val_data['regimes'], val_data['prices'],
            val_data['atrs'], config
        )
        
        agent = SACAdvanced(state_dim=env.state_dim, hidden_dim=hidden_dim,
                           lr=lr, gamma=gamma, tau=tau, device=device)
        
        try:
            agent.train(env=env, total_steps=25000, batch_size=256, start_steps=2000, log_interval=100000)
            metrics = agent.evaluate(env, n_episodes=5)
            
            score = metrics['sharpe']
            if metrics['mean_return'] > 0.02:
                score += 1.5
            if metrics['mean_return'] < -0.03:
                score -= 2.0
            if metrics['max_drawdown'] > 0.10:
                score -= 1.0
            
            return score
        except:
            return -10.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=60, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info(f"🎯 Mejores parámetros: {best_params}")
    
    # =========================================
    # ENTRENAR MODELO FINAL
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   FASE 4: ENTRENANDO MODELO FINAL")
    logger.info("=" * 60)
    
    combined_data = {k: np.vstack([train_data[k], val_data[k]]) if k != 'regimes' and k != 'prices' and k != 'atrs' 
                     else np.concatenate([train_data[k], val_data[k]]) for k in train_data.keys()}
    
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
        state_dim=final_env.state_dim, hidden_dim=best_params['hidden_dim'],
        lr=best_params['lr'], gamma=best_params['gamma'], tau=best_params['tau'], device=device
    )
    
    final_agent.train(env=final_env, total_steps=250000, batch_size=256, start_steps=10000, log_interval=50000)
    final_agent.save(str(root_dir / "artifacts/ultimate/sac_agent.pt"))
    
    # =========================================
    # EVALUACIÓN HOLDOUT
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("   📊 EVALUACIÓN FINAL EN HOLDOUT")
    logger.info("=" * 60)
    
    holdout_env = AdvancedTradingEnv(
        holdout_data['embeddings'], holdout_data['pred_4h'], holdout_data['pred_12h'],
        holdout_data['pred_24h'], holdout_data['regimes'], holdout_data['prices'],
        holdout_data['atrs'], final_config
    )
    
    metrics = final_agent.evaluate(holdout_env, n_episodes=20)
    
    # =========================================
    # RESUMEN
    # =========================================
    print("\n" + "=" * 70)
    print("              ✅ ENTRENAMIENTO ULTIMATE COMPLETADO")
    print("=" * 70)
    
    print(f"\n📊 RESULTADOS EN HOLDOUT:")
    print(f"   Retorno medio: {metrics['mean_return']*100:+.2f}%")
    print(f"   Sharpe ratio: {metrics['sharpe']:.2f}")
    print(f"   Win rate: {metrics['mean_win_rate']*100:.1f}%")
    print(f"   Max drawdown: {metrics['max_drawdown']*100:.1f}%")
    print(f"   Trades promedio: {metrics['mean_trades']:.0f}")
    
    print(f"\n🎯 Mejores hiperparámetros:")
    for k, v in best_params.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    
    # Evaluación
    if metrics['sharpe'] > 1.5 and metrics['mean_return'] > 0.03:
        print("\n🎉 ¡EXCELENTE! Modelo listo para paper trading.")
    elif metrics['sharpe'] > 0.5 and metrics['mean_return'] > 0:
        print("\n✅ Modelo aceptable. Considerar más optimización.")
    else:
        print("\n⚠️ Modelo necesita más trabajo.")


if __name__ == "__main__":
    main()
