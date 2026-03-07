"""
Contrastive Learning para Series Temporales Financieras.
Pre-entrena representaciones de mercado sin labels.

Idea: Ventanas temporales cercanas son "similares",
      Ventanas lejanas o de diferente régimen son "diferentes".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from loguru import logger
from dataclasses import dataclass


@dataclass
class ContrastiveConfig:
    """Configuración para Contrastive Learning."""
    seq_len: int = 50
    embedding_dim: int = 128
    hidden_dim: int = 256
    temperature: float = 0.07
    augmentation_strength: float = 0.1
    num_negatives: int = 64


class TimeSeriesAugmenter:
    """Augmentaciones para series temporales financieras."""
    
    def __init__(self, strength: float = 0.1):
        self.strength = strength
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica augmentaciones aleatorias."""
        aug_funcs = [
            self.jitter,
            self.scaling,
            self.time_warp,
            self.magnitude_warp,
        ]
        
        # Aplicar 1-2 augmentaciones aleatorias
        n_augs = np.random.randint(1, 3)
        selected = np.random.choice(len(aug_funcs), n_augs, replace=False)
        
        x_aug = x.clone()
        for idx in selected:
            x_aug = aug_funcs[idx](x_aug)
        
        return x_aug
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Añade ruido gaussiano."""
        noise = torch.randn_like(x) * self.strength * x.std()
        return x + noise
    
    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Escala aleatoria."""
        scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.strength
        return x * scale
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Distorsión temporal suave."""
        # Simplificado: shift aleatorio pequeño
        shift = np.random.randint(-2, 3)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-2)
    
    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Distorsión de magnitud."""
        seq_len = x.shape[-2]
        warp = torch.linspace(1 - self.strength, 1 + self.strength, seq_len)
        warp = warp.view(1, -1, 1).to(x.device)
        return x * warp


class ContrastiveEncoder(nn.Module):
    """Encoder para Contrastive Learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Projection head (importante para contrastive learning)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Representation head (para downstream tasks)
        self.representer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor, return_projection: bool = True) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            return_projection: Si True, retorna para contrastive loss
                              Si False, retorna representación para downstream
        """
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        if return_projection:
            return self.projector(context)
        else:
            return self.representer(context)


class ContrastiveDataset(torch.utils.data.Dataset):
    """Dataset para Contrastive Learning."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 50,
        feature_columns: List[str] = None
    ):
        self.seq_len = seq_len
        
        if feature_columns is None:
            feature_columns = [
                'return_1', 'return_4', 'return_24',
                'rsi', 'macd_hist', 'bb_position', 'atr_pct',
                'adx', 'di_plus', 'di_minus',
                'vol_ratio', 'trend_ema', 'trend_long', 'trend_strength'
            ]
        
        available = [c for c in feature_columns if c in df.columns]
        self.data = df[available].values.astype(np.float32)
        
        # Normalización
        self.mean = np.nanmean(self.data, axis=0)
        self.std = np.nanstd(self.data, axis=0) + 1e-8
        self.data = (self.data - self.mean) / self.std
        self.data = np.nan_to_num(self.data, 0)
        
        self.valid_indices = list(range(seq_len, len(self.data) - seq_len))
        
        logger.info(f"📊 ContrastiveDataset: {len(self.valid_indices)} samples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        center = self.valid_indices[idx]
        
        # Anchor
        anchor = self.data[center - self.seq_len:center]
        
        # Positive: ventana cercana (±5 pasos)
        pos_offset = np.random.randint(-5, 6)
        pos_center = np.clip(center + pos_offset, self.seq_len, len(self.data) - 1)
        positive = self.data[pos_center - self.seq_len:pos_center]
        
        return {
            'anchor': torch.tensor(anchor, dtype=torch.float32),
            'positive': torch.tensor(positive, dtype=torch.float32)
        }


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (batch, embedding_dim) - anchor embeddings
            z_j: (batch, embedding_dim) - positive embeddings
        """
        batch_size = z_i.shape[0]
        
        # Normalizar
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenar
        z = torch.cat([z_i, z_j], dim=0)
        
        # Matriz de similitud
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Máscara para positivos
        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Máscara para negativos (excluir diagonal)
        mask = (~torch.eye(2 * batch_size, dtype=bool, device=z.device)).float()
        
        # Numerador: exp de positivos
        numerator = torch.exp(positives)
        
        # Denominador: suma de exp de todos (excepto self)
        denominator = (mask * torch.exp(sim)).sum(dim=1)
        
        # Loss
        loss = -torch.log(numerator / denominator).mean()
        
        return loss


class ContrastiveTrainer:
    """Entrenador para Contrastive Learning."""
    
    def __init__(
        self,
        encoder: ContrastiveEncoder,
        config: ContrastiveConfig,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)
        self.config = config
        
        self.augmenter = TimeSeriesAugmenter(config.augmentation_strength)
        self.criterion = NTXentLoss(config.temperature)
        
        self.optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        logger.info(f"🏋️ ContrastiveTrainer en {self.device}")
    
    def train_epoch(self, dataloader) -> float:
        self.encoder.train()
        total_loss = 0
        
        for batch in dataloader:
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            
            # Augmentaciones
            anchor_aug = self.augmenter(anchor)
            positive_aug = self.augmenter(positive)
            
            # Forward
            z_anchor = self.encoder(anchor_aug, return_projection=True)
            z_positive = self.encoder(positive_aug, return_projection=True)
            
            # Loss
            loss = self.criterion(z_anchor, z_positive)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def train(
        self,
        dataloader,
        epochs: int = 100,
        log_interval: int = 10
    ) -> List[float]:
        logger.info(f"🚀 Pre-entrenamiento contrastivo: {epochs} épocas")
        
        losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            loss = self.train_epoch(dataloader)
            losses.append(loss)
            
            if loss < best_loss:
                best_loss = loss
                self.best_state = self.encoder.state_dict().copy()
            
            if (epoch + 1) % log_interval == 0:
                logger.info(f"Época {epoch+1}/{epochs} | Loss: {loss:.4f}")
        
        self.encoder.load_state_dict(self.best_state)
        logger.info(f"✅ Pre-entrenamiento completado. Best loss: {best_loss:.4f}")
        
        return losses
    
    def get_encoder(self) -> ContrastiveEncoder:
        """Retorna encoder entrenado para downstream tasks."""
        return self.encoder
    
    def save(self, path: str):
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"💾 Encoder guardado: {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        logger.info(f"📂 Encoder cargado: {path}")


def test_contrastive():
    """Test del sistema contrastivo."""
    print("🧪 Test Contrastive Learning...")
    
    # Config
    config = ContrastiveConfig()
    
    # Encoder
    encoder = ContrastiveEncoder(
        input_dim=14,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim
    )
    
    # Datos dummy
    batch_size = 32
    x = torch.randn(batch_size, 50, 14)
    
    # Forward
    z_proj = encoder(x, return_projection=True)
    z_repr = encoder(x, return_projection=False)
    
    print(f"✅ Projection shape: {z_proj.shape}")
    print(f"✅ Representation shape: {z_repr.shape}")
    
    # Loss
    criterion = NTXentLoss()
    loss = criterion(z_proj, z_proj)  # Self-similarity
    print(f"✅ Loss (self): {loss:.4f}")
    
    # Parámetros
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"📊 Parámetros: {total_params:,}")
    
    print("✅ Test completado")


if __name__ == "__main__":
    test_contrastive()
