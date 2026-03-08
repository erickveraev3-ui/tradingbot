from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GRUAlphaConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    regime_classes: int = 4


class GRUAlphaModel(nn.Module):
    """
    Modelo causal para trading de BTC con contexto crypto.

    Entradas:
    - Secuencia [batch, seq_len, input_dim]

    Salidas:
    - predicción retorno 1h
    - predicción retorno 4h
    - logits de dirección
    - logits de régimen
    """

    def __init__(self, config: GRUAlphaConfig):
        super().__init__()
        self.config = config

        # Normalización por feature en la última dimensión
        self.input_norm = nn.LayerNorm(config.input_dim)

        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.backbone = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Cabezas separadas
        self.return_head = nn.Linear(config.hidden_dim, 2)        # [ret_1h, ret_4h]
        self.direction_head = nn.Linear(config.hidden_dim, 2)     # [short_bias, long_bias]
        self.regime_head = nn.Linear(config.hidden_dim, config.regime_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: [batch, seq_len, input_dim]
        """
        x = self.input_norm(x)

        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]   # solo último paso temporal

        h = self.backbone(last_hidden)

        return_preds = self.return_head(h)
        direction_logits = self.direction_head(h)
        regime_logits = self.regime_head(h)

        return {
            "return_preds": return_preds,           # shape [B, 2]
            "direction_logits": direction_logits,   # shape [B, 2]
            "regime_logits": regime_logits,         # shape [B, C]
            "embedding": h,                         # útil para análisis/debug
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(x)

        direction_prob = torch.softmax(out["direction_logits"], dim=-1)
        regime_prob = torch.softmax(out["regime_logits"], dim=-1)

        return {
            **out,
            "direction_prob": direction_prob,
            "regime_prob": regime_prob,
        }


def build_model(input_dim: int) -> GRUAlphaModel:
    config = GRUAlphaConfig(input_dim=input_dim)
    return GRUAlphaModel(config)
