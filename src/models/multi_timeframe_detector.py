"""
Detector Multi-Timeframe con Detección de Régimen.
Combina 3 temporalidades + clasifica el régimen de mercado.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from loguru import logger
from enum import IntEnum


class MarketRegime(IntEnum):
    """Regímenes de mercado."""
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    HIGH_VOLATILITY = 3


class TimeframeLSTM(nn.Module):
    """LSTM para una temporalidad específica."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        self.norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            embedding: (batch, embedding_dim)
        """
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Project to embedding
        embedding = self.projection(context)
        
        return embedding


class RegimeDetector(nn.Module):
    """Clasificador de régimen de mercado."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 regímenes
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            regime_logits: (batch, 4)
            regime_probs: (batch, 4)
        """
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class MultiTimeframeDetector(nn.Module):
    """
    Detector que combina 3 temporalidades + detección de régimen.
    
    Temporalidades:
    - Daily (1D): 30 velas = 30 días → Tendencia macro
    - 4 Hour (4H): 50 velas = 8.3 días → Swing trading
    - 1 Hour (1H): 24 velas = 1 día → Timing de entrada
    """
    
    def __init__(
        self,
        input_dim_1d: int = 14,
        input_dim_4h: int = 14,
        input_dim_1h: int = 14,
        hidden_dim: int = 128,
        embedding_dim_1d: int = 64,
        embedding_dim_4h: int = 64,
        embedding_dim_1h: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim_total = embedding_dim_1d + embedding_dim_4h + embedding_dim_1h
        
        # LSTM para cada timeframe
        self.lstm_1d = TimeframeLSTM(
            input_dim=input_dim_1d,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim_1d,
            dropout=dropout
        )
        
        self.lstm_4h = TimeframeLSTM(
            input_dim=input_dim_4h,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim_4h,
            dropout=dropout
        )
        
        self.lstm_1h = TimeframeLSTM(
            input_dim=input_dim_1h,
            hidden_dim=hidden_dim // 2,
            embedding_dim=embedding_dim_1h,
            dropout=dropout
        )
        
        # Detector de régimen
        self.regime_detector = RegimeDetector(
            input_dim=self.embedding_dim_total,
            hidden_dim=64
        )
        
        # Predictor de retornos (para cada timeframe)
        self.predictor_4h = nn.Sequential(
            nn.Linear(self.embedding_dim_total, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # q10, q50, q90 para 4H
        )
        
        self.predictor_12h = nn.Sequential(
            nn.Linear(self.embedding_dim_total, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # q10, q50, q90 para 12H
        )
        
        self.predictor_24h = nn.Sequential(
            nn.Linear(self.embedding_dim_total, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # q10, q50, q90 para 24H
        )
        
        logger.info(f"🧠 MultiTimeframeDetector creado:")
        logger.info(f"   1D embedding: {embedding_dim_1d}")
        logger.info(f"   4H embedding: {embedding_dim_4h}")
        logger.info(f"   1H embedding: {embedding_dim_1h}")
        logger.info(f"   Total embedding: {self.embedding_dim_total}")
    
    def forward(
        self,
        x_1d: torch.Tensor,
        x_4h: torch.Tensor,
        x_1h: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_1d: (batch, 30, input_dim) - 30 días
            x_4h: (batch, 50, input_dim) - 50 velas 4H
            x_1h: (batch, 24, input_dim) - 24 horas
            
        Returns:
            dict con embeddings, predicciones y régimen
        """
        # Embeddings por timeframe
        emb_1d = self.lstm_1d(x_1d)
        emb_4h = self.lstm_4h(x_4h)
        emb_1h = self.lstm_1h(x_1h)
        
        # Concatenar embeddings
        combined = torch.cat([emb_1d, emb_4h, emb_1h], dim=-1)
        
        # Detectar régimen
        regime_logits, regime_probs = self.regime_detector(combined)
        
        # Predicciones de retorno
        pred_4h = self.predictor_4h(combined)
        pred_12h = self.predictor_12h(combined)
        pred_24h = self.predictor_24h(combined)
        
        return {
            'embedding': combined,
            'emb_1d': emb_1d,
            'emb_4h': emb_4h,
            'emb_1h': emb_1h,
            'regime_logits': regime_logits,
            'regime_probs': regime_probs,
            'pred_4h': pred_4h,
            'pred_12h': pred_12h,
            'pred_24h': pred_24h
        }


class MultiTimeframeDataset(torch.utils.data.Dataset):
    """Dataset que combina las 3 temporalidades."""
    
    def __init__(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        seq_len_1d: int = 30,
        seq_len_4h: int = 50,
        seq_len_1h: int = 24,
        horizon_4h: int = 1,    # 4 horas
        horizon_12h: int = 3,   # 12 horas (3 velas de 4H)
        horizon_24h: int = 6,   # 24 horas (6 velas de 4H)
        feature_columns: List[str] = None
    ):
        self.seq_len_1d = seq_len_1d
        self.seq_len_4h = seq_len_4h
        self.seq_len_1h = seq_len_1h
        
        # Features por defecto
        if feature_columns is None:
            feature_columns = [
                'return_1', 'return_4', 'return_24',
                'rsi', 'macd_hist', 'bb_position', 'atr_pct',
                'adx', 'di_plus', 'di_minus',
                'vol_ratio', 'trend_ema', 'trend_long', 'trend_strength'
            ]
        
        # Filtrar columnas disponibles para cada timeframe
        self.features_1d = [c for c in feature_columns if c in df_1d.columns]
        self.features_4h = [c for c in feature_columns if c in df_4h.columns]
        self.features_1h = [c for c in feature_columns if c in df_1h.columns]
        
        # Preparar datos
        self.data_1d = self._prepare_data(df_1d, self.features_1d)
        self.data_4h = self._prepare_data(df_4h, self.features_4h)
        self.data_1h = self._prepare_data(df_1h, self.features_1h)
        
        # Timestamps para alineación
        self.timestamps_4h = pd.to_datetime(df_4h['timestamp']).values
        self.timestamps_1h = pd.to_datetime(df_1h['timestamp']).values
        self.timestamps_1d = pd.to_datetime(df_1d['timestamp']).values
        
        # Precios para calcular retornos reales
        self.prices_4h = df_4h['close'].values
        
        # Calcular retornos futuros (targets)
        self.returns_4h = self._calc_future_returns(self.prices_4h, horizon_4h)
        self.returns_12h = self._calc_future_returns(self.prices_4h, horizon_12h)
        self.returns_24h = self._calc_future_returns(self.prices_4h, horizon_24h)
        
        # Calcular régimen de mercado (target)
        self.regimes = self._calc_regimes(df_4h)
        
        # Índices válidos (donde tenemos todas las temporalidades)
        self._calc_valid_indices()
        
        logger.info(f"📊 MultiTimeframeDataset creado:")
        logger.info(f"   1D: {len(self.data_1d)} velas, {len(self.features_1d)} features")
        logger.info(f"   4H: {len(self.data_4h)} velas, {len(self.features_4h)} features")
        logger.info(f"   1H: {len(self.data_1h)} velas, {len(self.features_1h)} features")
        logger.info(f"   Samples válidos: {len(self.valid_indices)}")
    
    def _prepare_data(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """Prepara y normaliza datos."""
        data = df[features].values.astype(np.float32)
        
        # Z-score normalization
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0) + 1e-8
        data = (data - mean) / std
        
        # Reemplazar NaN
        data = np.nan_to_num(data, 0)
        
        return data
    
    def _calc_future_returns(self, prices: np.ndarray, horizon: int) -> np.ndarray:
        """Calcula retornos futuros."""
        returns = np.zeros(len(prices))
        for i in range(len(prices) - horizon):
            returns[i] = (prices[i + horizon] - prices[i]) / prices[i]
        return returns.astype(np.float32)
    
    def _calc_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula el régimen de mercado para cada punto.
        0: Tendencia alcista
        1: Tendencia bajista
        2: Rango/Lateral
        3: Alta volatilidad
        """
        regimes = np.zeros(len(df), dtype=np.int64)
        
        # Calcular métricas
        returns = df['close'].pct_change(20).values  # Retorno 20 períodos
        volatility = df['close'].pct_change().rolling(20).std().values
        adx = df['adx'].values if 'adx' in df.columns else np.zeros(len(df))
        
        vol_threshold = np.nanpercentile(volatility, 75)
        
        for i in range(len(df)):
            if np.isnan(returns[i]) or np.isnan(volatility[i]):
                regimes[i] = 2  # Default: ranging
                continue
            
            # Alta volatilidad
            if volatility[i] > vol_threshold:
                regimes[i] = 3
            # Tendencia alcista
            elif returns[i] > 0.02 and adx[i] > 25:
                regimes[i] = 0
            # Tendencia bajista
            elif returns[i] < -0.02 and adx[i] > 25:
                regimes[i] = 1
            # Rango
            else:
                regimes[i] = 2
        
        return regimes
    
    def _calc_valid_indices(self):
        """Calcula índices donde tenemos datos de todas las temporalidades."""
        self.valid_indices = []
        self.alignment_map = []  # Mapea idx_4h -> (idx_1d, idx_1h_start)
        
        for idx_4h in range(self.seq_len_4h, len(self.data_4h) - 6):
            ts_4h = self.timestamps_4h[idx_4h]
            
            # Buscar índice correspondiente en 1D
            ts_1d_needed = ts_4h - pd.Timedelta(days=self.seq_len_1d)
            mask_1d = self.timestamps_1d <= ts_4h
            if mask_1d.sum() < self.seq_len_1d:
                continue
            idx_1d = mask_1d.sum() - 1
            
            # Buscar índice correspondiente en 1H
            mask_1h = self.timestamps_1h <= ts_4h
            if mask_1h.sum() < self.seq_len_1h:
                continue
            idx_1h = mask_1h.sum() - 1
            
            self.valid_indices.append(idx_4h)
            self.alignment_map.append((idx_1d, idx_1h))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        idx_4h = self.valid_indices[idx]
        idx_1d, idx_1h = self.alignment_map[idx]
        
        # Extraer secuencias
        x_1d = self.data_1d[idx_1d - self.seq_len_1d + 1:idx_1d + 1]
        x_4h = self.data_4h[idx_4h - self.seq_len_4h + 1:idx_4h + 1]
        x_1h = self.data_1h[idx_1h - self.seq_len_1h + 1:idx_1h + 1]
        
        # Targets
        ret_4h = self.returns_4h[idx_4h]
        ret_12h = self.returns_12h[idx_4h]
        ret_24h = self.returns_24h[idx_4h]
        regime = self.regimes[idx_4h]
        
        return {
            'x_1d': torch.tensor(x_1d, dtype=torch.float32),
            'x_4h': torch.tensor(x_4h, dtype=torch.float32),
            'x_1h': torch.tensor(x_1h, dtype=torch.float32),
            'ret_4h': torch.tensor([ret_4h], dtype=torch.float32),
            'ret_12h': torch.tensor([ret_12h], dtype=torch.float32),
            'ret_24h': torch.tensor([ret_24h], dtype=torch.float32),
            'regime': torch.tensor(regime, dtype=torch.long),
            'price': torch.tensor(self.prices_4h[idx_4h], dtype=torch.float32)
        }


class MultiTimeframeTrainer:
    """Entrenador para el detector multi-timeframe."""
    
    def __init__(
        self,
        model: MultiTimeframeDetector,
        lr: float = 1e-3,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        logger.info(f"🏋️ MultiTimeframeTrainer en {self.device}")
    
    def quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pinball loss para predicción de cuantiles."""
        quantiles = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        losses = []
        
        for i, q in enumerate(quantiles):
            error = target - pred[:, i:i+1]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_regime_loss = 0
        total_pred_loss = 0
        
        for batch in dataloader:
            x_1d = batch['x_1d'].to(self.device)
            x_4h = batch['x_4h'].to(self.device)
            x_1h = batch['x_1h'].to(self.device)
            ret_4h = batch['ret_4h'].to(self.device)
            ret_12h = batch['ret_12h'].to(self.device)
            ret_24h = batch['ret_24h'].to(self.device)
            regime = batch['regime'].to(self.device)
            
            # Forward
            outputs = self.model(x_1d, x_4h, x_1h)
            
            # Losses
            loss_regime = nn.functional.cross_entropy(outputs['regime_logits'], regime)
            loss_4h = self.quantile_loss(outputs['pred_4h'], ret_4h)
            loss_12h = self.quantile_loss(outputs['pred_12h'], ret_12h)
            loss_24h = self.quantile_loss(outputs['pred_24h'], ret_24h)
            
            loss_pred = (loss_4h + loss_12h + loss_24h) / 3
            loss = loss_pred + 0.5 * loss_regime
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_regime_loss += loss_regime.item()
            total_pred_loss += loss_pred.item()
        
        n = len(dataloader)
        return {
            'total_loss': total_loss / n,
            'regime_loss': total_regime_loss / n,
            'pred_loss': total_pred_loss / n
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct_regime = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x_1d = batch['x_1d'].to(self.device)
                x_4h = batch['x_4h'].to(self.device)
                x_1h = batch['x_1h'].to(self.device)
                ret_12h = batch['ret_12h'].to(self.device)
                regime = batch['regime'].to(self.device)
                
                outputs = self.model(x_1d, x_4h, x_1h)
                
                loss = self.quantile_loss(outputs['pred_12h'], ret_12h)
                total_loss += loss.item()
                
                # Accuracy de régimen
                pred_regime = outputs['regime_logits'].argmax(dim=-1)
                correct_regime += (pred_regime == regime).sum().item()
                total_samples += len(regime)
        
        return {
            'val_loss': total_loss / len(dataloader),
            'regime_accuracy': correct_regime / total_samples
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping: int = 15
    ) -> Dict[str, List]:
        logger.info(f"🚀 Entrenamiento: {epochs} épocas")
        
        history = {'train_loss': [], 'val_loss': [], 'regime_acc': []}
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['regime_acc'].append(val_metrics['regime_accuracy'])
            
            self.scheduler.step(val_metrics['val_loss'])
            
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Época {epoch+1}/{epochs} | "
                    f"Train: {train_metrics['total_loss']:.4f} | "
                    f"Val: {val_metrics['val_loss']:.4f} | "
                    f"Regime Acc: {val_metrics['regime_accuracy']:.1%}"
                )
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience = 0
                self.best_state = self.model.state_dict().copy()
            else:
                patience += 1
                if patience >= early_stopping:
                    logger.info(f"⏹️ Early stopping época {epoch+1}")
                    break
        
        self.model.load_state_dict(self.best_state)
        logger.info(f"✅ Entrenamiento completado. Best val loss: {best_val_loss:.4f}")
        
        return history
    
    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'model_config': {
                'embedding_dim_total': self.model.embedding_dim_total
            }
        }, path)
        logger.info(f"💾 Modelo guardado: {path}")


def test_multi_timeframe():
    """Test del detector multi-timeframe."""
    print("🧪 Test MultiTimeframeDetector...")
    
    # Crear modelo
    model = MultiTimeframeDetector(
        input_dim_1d=14,
        input_dim_4h=14,
        input_dim_1h=14
    )
    
    # Datos dummy
    batch_size = 4
    x_1d = torch.randn(batch_size, 30, 14)
    x_4h = torch.randn(batch_size, 50, 14)
    x_1h = torch.randn(batch_size, 24, 14)
    
    # Forward
    outputs = model(x_1d, x_4h, x_1h)
    
    print(f"✅ Embedding shape: {outputs['embedding'].shape}")
    print(f"✅ Regime probs shape: {outputs['regime_probs'].shape}")
    print(f"✅ Pred 12h shape: {outputs['pred_12h'].shape}")
    
    # Parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parámetros totales: {total_params:,}")


if __name__ == "__main__":
    test_multi_timeframe()
