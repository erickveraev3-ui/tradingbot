"""
Sistema Completo de Trading con:
1. Contrastive Learning (pre-entrenamiento)
2. Chronos (foundation model)
3. Multi-Timeframe LSTM
4. SAC Advanced (decisiones)
5. Kelly + Trailing Stop (risk management)
"""
import sys
from pathlib import Path

# Añadir root al path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
# ... resto del código

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass

# Chronos
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    logger.warning("⚠️ Chronos no disponible, usando fallback")

from src.models.contrastive_pretrain import (
    ContrastiveEncoder,
    ContrastiveConfig,
    ContrastiveDataset,
    ContrastiveTrainer
)


@dataclass 
class CompleteSystemConfig:
    """Configuración del sistema completo."""
    # Contrastive
    contrastive_embedding_dim: int = 128
    contrastive_epochs: int = 50
    
    # Chronos
    use_chronos: bool = True
    chronos_model: str = "amazon/chronos-t5-small"
    chronos_prediction_length: int = 12  # 12 velas de 4H = 48 horas
    
    # Multi-timeframe
    seq_len_1d: int = 30
    seq_len_4h: int = 50
    seq_len_1h: int = 24
    
    # Fusion
    fusion_hidden_dim: int = 256
    fusion_output_dim: int = 128
    
    # Trading
    initial_capital: float = 10000.0
    max_leverage: int = 4
    kelly_fraction: float = 0.5
    atr_stop: float = 2.5
    atr_trail: float = 2.0
    max_drawdown: float = 0.18


class ChronosPredictor:
    """Wrapper para Chronos foundation model."""
    
    def __init__(self, model_name: str = "amazon/chronos-t5-small", device: str = "cuda"):
        self.device = device
        
        if CHRONOS_AVAILABLE:
            logger.info(f"📥 Cargando Chronos: {model_name}")
            self.pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float32
            )
            self.available = True
            logger.info("✅ Chronos cargado")
        else:
            self.available = False
            logger.warning("⚠️ Chronos no disponible")
    
    def predict(
        self, 
        prices: np.ndarray, 
        prediction_length: int = 12,
        num_samples: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Genera predicciones probabilísticas.
        
        Args:
            prices: Array de precios históricos
            prediction_length: Número de pasos a predecir
            num_samples: Número de muestras para cuantiles
            
        Returns:
            Dict con q10, q50, q90 de retornos predichos
        """
        if not self.available:
            # Fallback: predicción naive
            last_price = prices[-1]
            noise = np.random.randn(prediction_length) * 0.02
            pred_prices = last_price * (1 + np.cumsum(noise))
            returns = (pred_prices - last_price) / last_price
            return {
                'q10': returns * 0.8,
                'q50': returns,
                'q90': returns * 1.2
            }
        
        # Chronos prediction
        context = torch.tensor(prices[-200:]).unsqueeze(0)  # Último contexto
        
        forecast = self.pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples
        )
        
        # Calcular cuantiles
        forecast_np = forecast.numpy()[0]  # (num_samples, prediction_length)
        
        q10 = np.percentile(forecast_np, 10, axis=0)
        q50 = np.percentile(forecast_np, 50, axis=0)
        q90 = np.percentile(forecast_np, 90, axis=0)
        
        # Convertir a retornos
        last_price = prices[-1]
        returns_q10 = (q10 - last_price) / last_price
        returns_q50 = (q50 - last_price) / last_price
        returns_q90 = (q90 - last_price) / last_price
        
        return {
            'q10': returns_q10,
            'q50': returns_q50,
            'q90': returns_q90,
            'prices_q50': q50
        }


class MultiTimeframeLSTM(nn.Module):
    """LSTM Multi-Timeframe mejorado con Contrastive pre-training."""
    
    def __init__(
        self,
        input_dim: int,
        contrastive_encoder: Optional[ContrastiveEncoder] = None,
        hidden_dim: int = 128,
        output_dim: int = 64
    ):
        super().__init__()
        
        self.use_pretrained = contrastive_encoder is not None
        
        if self.use_pretrained:
            self.encoder = contrastive_encoder
            # Congelar parcialmente (fine-tune solo últimas capas)
            for param in self.encoder.lstm.parameters():
                param.requires_grad = False
            encoder_dim = contrastive_encoder.representer[-1].out_features
        else:
            self.encoder = None
            encoder_dim = 0
        
        # LSTM adicional para capturar patrones específicos
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Combinar encoder pretrained + LSTM
        combined_dim = encoder_dim + hidden_dim * 2 if self.use_pretrained else hidden_dim * 2
        
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            embedding: (batch, output_dim)
        """
        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        lstm_emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        if self.use_pretrained:
            # Contrastive embedding
            with torch.no_grad():
                contra_emb = self.encoder(x, return_projection=False)
            combined = torch.cat([contra_emb, lstm_emb], dim=-1)
        else:
            combined = lstm_emb
        
        return self.output_proj(combined)


class FusionNetwork(nn.Module):
    """
    Red de fusión que combina:
    - Multi-timeframe embeddings (1H, 4H, 1D)
    - Chronos predictions
    - Regime detection
    """
    
    def __init__(
        self,
        input_dim_1h: int,
        input_dim_4h: int,
        input_dim_1d: int,
        chronos_dim: int = 36,  # 12 steps * 3 quantiles
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()
        
        # LSTMs por timeframe
        self.lstm_1h = MultiTimeframeLSTM(input_dim_1h, output_dim=48)
        self.lstm_4h = MultiTimeframeLSTM(input_dim_4h, output_dim=64)
        self.lstm_1d = MultiTimeframeLSTM(input_dim_1d, output_dim=64)
        
        # Total: 48 + 64 + 64 + 36 (chronos) = 212
        fusion_input_dim = 48 + 64 + 64 + chronos_dim
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Regime classifier
        self.regime_head = nn.Linear(hidden_dim, 4)
        
        # Return predictor (3 horizontes × 3 cuantiles)
        self.return_head = nn.Linear(hidden_dim, 9)
        
        # Final embedding
        self.embedding_head = nn.Linear(hidden_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(
        self,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        x_1d: torch.Tensor,
        chronos_preds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_1h: (batch, 24, features)
            x_4h: (batch, 50, features)
            x_1d: (batch, 30, features)
            chronos_preds: (batch, 36) - flattened chronos predictions
        """
        # Embeddings por timeframe
        emb_1h = self.lstm_1h(x_1h)
        emb_4h = self.lstm_4h(x_4h)
        emb_1d = self.lstm_1d(x_1d)
        
        # Concatenar todo
        combined = torch.cat([emb_1h, emb_4h, emb_1d, chronos_preds], dim=-1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Outputs
        regime_logits = self.regime_head(fused)
        return_preds = self.return_head(fused)
        embedding = self.embedding_head(fused)
        
        return {
            'embedding': embedding,
            'regime_logits': regime_logits,
            'regime_probs': torch.softmax(regime_logits, dim=-1),
            'return_preds': return_preds.view(-1, 3, 3),  # (batch, 3 horizontes, 3 cuantiles)
        }


class CompleteSystem:
    """
    Sistema completo de trading.
    Orquesta todos los componentes.
    """
    
    def __init__(self, config: CompleteSystemConfig = None, device: str = None):
        self.config = config or CompleteSystemConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("🚀 Inicializando Sistema Completo de Trading")
        logger.info(f"   Device: {self.device}")
        
        # Componentes (se inicializan en build())
        self.contrastive_encoder = None
        self.chronos = None
        self.fusion_network = None
        self.sac_agent = None
        
        self.is_built = False
    
    def pretrain_contrastive(self, df_4h: pd.DataFrame) -> ContrastiveEncoder:
        """Fase 1: Pre-entrenar con Contrastive Learning."""
        logger.info("\n" + "=" * 60)
        logger.info("   FASE 1: PRE-ENTRENAMIENTO CONTRASTIVO")
        logger.info("=" * 60)
        
        # Preparar datos
        from src.features.indicators import calculate_all_indicators
        df = calculate_all_indicators(df_4h.copy())
        
        dataset = ContrastiveDataset(df, seq_len=50)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True, 
            num_workers=4, pin_memory=True, drop_last=True
        )
        
        # Crear encoder
        sample = dataset[0]['anchor']
        input_dim = sample.shape[-1]
        
        config = ContrastiveConfig(
            embedding_dim=self.config.contrastive_embedding_dim,
            temperature=0.07
        )
        
        encoder = ContrastiveEncoder(
            input_dim=input_dim,
            hidden_dim=256,
            embedding_dim=config.embedding_dim
        )
        
        # Entrenar
        trainer = ContrastiveTrainer(encoder, config, self.device)
        losses = trainer.train(dataloader, epochs=self.config.contrastive_epochs)
        
        self.contrastive_encoder = trainer.get_encoder()
        
        # Guardar
        save_path = Path("artifacts/contrastive/encoder.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(save_path))
        
        return self.contrastive_encoder
    
    def init_chronos(self):
        """Fase 2: Inicializar Chronos."""
        logger.info("\n" + "=" * 60)
        logger.info("   FASE 2: INICIALIZANDO CHRONOS")
        logger.info("=" * 60)
        
        if self.config.use_chronos:
            self.chronos = ChronosPredictor(
                self.config.chronos_model,
                self.device
            )
        else:
            self.chronos = ChronosPredictor.__new__(ChronosPredictor)
            self.chronos.available = False
    
    def build_fusion_network(self, input_dims: Dict[str, int]):
        """Fase 3: Construir red de fusión."""
        logger.info("\n" + "=" * 60)
        logger.info("   FASE 3: CONSTRUYENDO RED DE FUSIÓN")
        logger.info("=" * 60)
        
        self.fusion_network = FusionNetwork(
            input_dim_1h=input_dims['1h'],
            input_dim_4h=input_dims['4h'],
            input_dim_1d=input_dims['1d'],
            chronos_dim=36,
            hidden_dim=self.config.fusion_hidden_dim,
            output_dim=self.config.fusion_output_dim
        ).to(self.device)
        
        # Si tenemos encoder contrastivo, integrarlo
        if self.contrastive_encoder is not None:
            self.fusion_network.lstm_4h = MultiTimeframeLSTM(
                input_dims['4h'],
                contrastive_encoder=self.contrastive_encoder,
                output_dim=64
            ).to(self.device)
            logger.info("   ✅ Encoder contrastivo integrado en 4H LSTM")
        
        total_params = sum(p.numel() for p in self.fusion_network.parameters())
        logger.info(f"   📊 Parámetros fusión: {total_params:,}")
        
        self.is_built = True
    
    def get_chronos_features(self, prices: np.ndarray) -> np.ndarray:
        """Obtiene features de Chronos."""
        if self.chronos is None or not self.chronos.available:
            # Fallback
            return np.zeros(36, dtype=np.float32)
        
        preds = self.chronos.predict(prices, prediction_length=12)
        
        # Flatten: 3 cuantiles × 12 steps = 36
        features = np.concatenate([
            preds['q10'],
            preds['q50'],
            preds['q90']
        ])
        
        return features.astype(np.float32)
    
    def forward(
        self,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        x_1d: torch.Tensor,
        prices_4h: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Forward pass completo."""
        
        if not self.is_built:
            raise RuntimeError("Sistema no construido. Llama build_fusion_network() primero.")
        
        batch_size = x_4h.shape[0]
        
        # Chronos features para cada sample del batch
        chronos_features = []
        for i in range(batch_size):
            # Usar precios hasta el punto actual
            cf = self.get_chronos_features(prices_4h[:i+100] if i+100 < len(prices_4h) else prices_4h)
            chronos_features.append(cf)
        
        chronos_tensor = torch.tensor(
            np.stack(chronos_features),
            dtype=torch.float32,
            device=self.device
        )
        
        # Forward fusion
        outputs = self.fusion_network(x_1h, x_4h, x_1d, chronos_tensor)
        
        return outputs
    
    def save(self, path: str):
        """Guarda el sistema completo."""
        save_dict = {
            'config': self.config,
            'fusion_state': self.fusion_network.state_dict() if self.fusion_network else None,
        }
        torch.save(save_dict, path)
        logger.info(f"💾 Sistema guardado: {path}")
    
    def load(self, path: str):
        """Carga el sistema."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        if checkpoint['fusion_state'] and self.fusion_network:
            self.fusion_network.load_state_dict(checkpoint['fusion_state'])
        logger.info(f"📂 Sistema cargado: {path}")


def test_complete_system():
    """Test del sistema completo."""
    print("🧪 Test Sistema Completo...")
    
    config = CompleteSystemConfig(
        use_chronos=CHRONOS_AVAILABLE,
        contrastive_epochs=2  # Solo para test
    )
    
    system = CompleteSystem(config)
    
    # Test fusion network (sin pre-entrenamiento para test rápido)
    system.build_fusion_network({
        '1h': 14,
        '4h': 14,
        '1d': 14
    })
    
    # Datos dummy
    batch_size = 4
    x_1h = torch.randn(batch_size, 24, 14).to(system.device)
    x_4h = torch.randn(batch_size, 50, 14).to(system.device)
    x_1d = torch.randn(batch_size, 30, 14).to(system.device)
    prices = np.random.randn(500) * 1000 + 50000
    
    # Forward
    outputs = system.forward(x_1h, x_4h, x_1d, prices)
    
    print(f"✅ Embedding shape: {outputs['embedding'].shape}")
    print(f"✅ Regime probs shape: {outputs['regime_probs'].shape}")
    print(f"✅ Return preds shape: {outputs['return_preds'].shape}")
    
    print("✅ Test completado")


if __name__ == "__main__":
    test_complete_system()
