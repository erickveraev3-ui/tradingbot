"""
Detector de Patrones con LSTM.
Aprende representaciones del mercado sin reglas predefinidas.
Input: Secuencia de velas con indicadores
Output: Embedding (vector que representa el estado del mercado)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger


class PatternDetectorLSTM(nn.Module):
    """
    Red LSTM que aprende a detectar patrones en el mercado.
    
    No le decimos qué patrones buscar - ella los descubre.
    El embedding de salida captura el "estado" del mercado.
    """
    
    def __init__(
        self,
        input_dim: int = 20,          # Número de features por vela
        hidden_dim: int = 128,         # Tamaño de la capa oculta LSTM
        num_layers: int = 2,           # Capas LSTM apiladas
        embedding_dim: int = 64,       # Dimensión del embedding de salida
        dropout: float = 0.2,          # Dropout para regularización
        bidirectional: bool = True     # LSTM bidireccional (ve pasado y "contexto")
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Normalización de entrada
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM principal
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Atención para ponderar qué velas son más importantes
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Proyección al embedding final
        self.fc_embedding = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Inicialización de pesos
        self._init_weights()
        
        logger.info(f"🧠 PatternDetector creado:")
        logger.info(f"   Input: {input_dim} features")
        logger.info(f"   LSTM: {num_layers} capas x {hidden_dim} hidden")
        logger.info(f"   Output embedding: {embedding_dim} dims")
    
    def _init_weights(self):
        """Inicializa pesos con Xavier/He."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Tensor de forma (batch, seq_len, input_dim)
            return_attention: Si True, retorna también los pesos de atención
            
        Returns:
            embedding: Tensor (batch, embedding_dim)
            attention_weights: Tensor (batch, seq_len) si return_attention=True
        """
        batch_size = x.size(0)
        
        # Normalizar entrada
        x = self.input_norm(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        
        # Calcular pesos de atención
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch, seq_len)
        
        # Aplicar atención (weighted sum)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_out                          # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden * num_directions)
        
        # Proyectar a embedding
        embedding = self.fc_embedding(context)  # (batch, embedding_dim)
        
        if return_attention:
            return embedding, attention_weights
        return embedding, None
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Obtiene solo el embedding (para inferencia)."""
        embedding, _ = self.forward(x)
        return embedding


class PatternDataset(torch.utils.data.Dataset):
    """
    Dataset para entrenar el detector de patrones.
    Crea secuencias de velas para el LSTM.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        seq_length: int = 50,           # Velas de contexto
        horizon: int = 4,               # Velas a predecir adelante
        feature_columns: list = None    # Columnas a usar como features
    ):
        """
        Args:
            df: DataFrame con indicadores calculados
            seq_length: Número de velas de contexto
            horizon: Horizonte de predicción (para el target)
            feature_columns: Lista de columnas a usar
        """
        self.seq_length = seq_length
        self.horizon = horizon
        
        # Features por defecto
        if feature_columns is None:
            feature_columns = [
                # Precio normalizado
                'return_1', 'return_4', 'return_24',
                # Indicadores
                'rsi', 'macd_hist', 'bb_position', 'atr_pct',
                'adx', 'di_plus', 'di_minus',
                'vol_ratio',
                # Señales
                'trend_ema', 'trend_long', 'trend_strength'
            ]
        
        # Filtrar columnas que existen
        self.feature_columns = [c for c in feature_columns if c in df.columns]
        logger.info(f"📊 Features usadas: {len(self.feature_columns)}")
        
        # Preparar datos
        self.data = df[self.feature_columns].values.astype(np.float32)
        
        # Target: retorno futuro (para entrenamiento supervisado)
        self.returns = df['close'].pct_change(horizon).shift(-horizon).values.astype(np.float32)
        
        # Normalizar features (z-score por columna)
        self.mean = np.nanmean(self.data, axis=0)
        self.std = np.nanstd(self.data, axis=0) + 1e-8
        self.data = (self.data - self.mean) / self.std
        
        # Reemplazar NaN con 0
        self.data = np.nan_to_num(self.data, 0)
        self.returns = np.nan_to_num(self.returns, 0)
        
        # Índices válidos (donde tenemos suficiente historia y futuro)
        self.valid_indices = list(range(seq_length, len(df) - horizon))
        
        logger.info(f"📊 Dataset creado: {len(self.valid_indices)} samples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Retorna una secuencia y su target.
        
        Returns:
            x: Tensor (seq_length, num_features)
            y: Tensor (1,) - retorno futuro
        """
        actual_idx = self.valid_indices[idx]
        
        # Secuencia de entrada
        x = self.data[actual_idx - self.seq_length:actual_idx]
        
        # Target: retorno futuro
        y = self.returns[actual_idx]
        
        return torch.tensor(x), torch.tensor([y])
    
    def get_normalizer_params(self) -> dict:
        """Retorna parámetros de normalización para usar en inferencia."""
        return {
            'mean': self.mean,
            'std': self.std,
            'feature_columns': self.feature_columns
        }


class PatternDetectorTrainer:
    """
    Entrenador para el detector de patrones.
    Usa predicción de retornos como tarea auxiliar para aprender buenos embeddings.
    """
    
    def __init__(
        self,
        model: PatternDetectorLSTM,
        learning_rate: float = 1e-3,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Cabeza de predicción (para entrenar los embeddings)
        self.predictor_head = nn.Sequential(
            nn.Linear(model.embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # q10, q50, q90
        ).to(self.device)
        
        # Optimizador
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.predictor_head.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        logger.info(f"🏋️ Trainer inicializado en {self.device}")
    
    def quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Pérdida de cuantiles (pinball loss).
        Entrena para predecir q10, q50, q90 del retorno futuro.
        """
        quantiles = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        losses = []
        
        for i, q in enumerate(quantiles):
            error = target - pred[:, i:i+1]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward
            embedding, _ = self.model(batch_x)
            pred = self.predictor_head(embedding)
            
            # Loss
            loss = self.quantile_loss(pred, batch_y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Valida el modelo."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                embedding, _ = self.model(batch_x)
                pred = self.predictor_head(embedding)
                loss = self.quantile_loss(pred, batch_y)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        early_stopping: int = 10
    ) -> dict:
        """
        Entrena el modelo completo.
        
        Returns:
            dict con historial de entrenamiento
        """
        logger.info(f"🚀 Iniciando entrenamiento: {epochs} épocas")
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Época {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar mejor modelo
                self.best_state = {
                    'model': self.model.state_dict(),
                    'predictor': self.predictor_head.state_dict()
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    logger.info(f"⏹️ Early stopping en época {epoch+1}")
                    break
        
        # Restaurar mejor modelo
        self.model.load_state_dict(self.best_state['model'])
        self.predictor_head.load_state_dict(self.best_state['predictor'])
        
        logger.info(f"✅ Entrenamiento completado. Best val loss: {best_val_loss:.6f}")
        
        return history
    
    def save(self, path: str):
        """Guarda el modelo."""
        torch.save({
            'model_state': self.model.state_dict(),
            'predictor_state': self.predictor_head.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'embedding_dim': self.model.embedding_dim,
                'bidirectional': self.model.bidirectional
            }
        }, path)
        logger.info(f"💾 Modelo guardado en: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None):
        """Carga un modelo guardado."""
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        
        # Recrear modelo
        config = checkpoint['model_config']
        model = PatternDetectorLSTM(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            embedding_dim=config['embedding_dim'],
            bidirectional=config['bidirectional']
        )
        model.load_state_dict(checkpoint['model_state'])
        
        trainer = cls(model, device=device)
        trainer.predictor_head.load_state_dict(checkpoint['predictor_state'])
        
        logger.info(f"📂 Modelo cargado desde: {path}")
        
        return trainer


def test_pattern_detector():
    """Prueba el detector de patrones."""
    import sys
    from pathlib import Path
    
    # Añadir el directorio raíz al path
    root_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root_dir))
    
    from src.features.indicators import calculate_all_indicators
    
    # Cargar datos
    data_path = root_dir / "data/raw/btcusdt_1h.csv"
    if not data_path.exists():
        print("❌ Primero ejecuta: python scripts/01_download_data.py")
        return
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calcular indicadores
    df = calculate_all_indicators(df)
    print(f"📊 Datos con indicadores: {len(df)} filas")
    
    # Crear dataset
    dataset = PatternDataset(df, seq_length=50, horizon=4)
    print(f"📊 Dataset: {len(dataset)} samples")
    
    # Crear modelo
    model = PatternDetectorLSTM(
        input_dim=len(dataset.feature_columns),
        hidden_dim=64,
        num_layers=2,
        embedding_dim=32
    )
    
    # Test forward pass
    sample_x, sample_y = dataset[0]
    sample_x = sample_x.unsqueeze(0)  # Añadir dimensión batch
    
    embedding, attention = model(sample_x, return_attention=True)
    
    print(f"\n🧠 Test forward pass:")
    print(f"   Input shape: {sample_x.shape}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Attention shape: {attention.shape}")
    print(f"   Target (retorno futuro): {sample_y.item()*100:.2f}%")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📈 Parámetros del modelo: {total_params:,}")


if __name__ == "__main__":
    test_pattern_detector()