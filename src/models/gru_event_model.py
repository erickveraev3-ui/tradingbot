from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GRUEventConfig:
    input_dim: int
    hidden_dim: int = 160
    num_layers: int = 2
    dropout: float = 0.20
    n_event_outputs: int = 4
    n_regime_classes: int = 4


class AttentionPooling(nn.Module):
    """
    Attention pooling causal sobre la secuencia ya procesada por la GRU.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, H]
        attn_logits = self.score(x)                  # [B, T, 1]
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (x * attn_weights).sum(dim=1)       # [B, H]
        return pooled, attn_weights.squeeze(-1)


class GRUEventModel(nn.Module):
    """
    Modelo secuencial para eventos de trading.

    Salidas:
    - event_logits: multi-label [long_event, short_event, breakout_up, breakdown_down]
    - regime_logits: clasificación auxiliar del régimen
    """

    def __init__(self, config: GRUEventConfig):
        super().__init__()
        self.config = config

        self.input_norm = nn.LayerNorm(config.input_dim)

        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.attn_pool = AttentionPooling(config.hidden_dim)

        self.backbone = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.event_head = nn.Linear(config.hidden_dim, config.n_event_outputs)
        self.regime_head = nn.Linear(config.hidden_dim, config.n_regime_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: [B, T, F]
        """
        x = self.input_norm(x)

        seq_out, _ = self.gru(x)                 # [B, T, H]
        last_hidden = seq_out[:, -1, :]          # [B, H]
        pooled, attn_weights = self.attn_pool(seq_out)

        # fusionamos último estado + attention pooling
        h = torch.cat([last_hidden, pooled], dim=-1)
        h = self.backbone(h)

        event_logits = self.event_head(h)        # [B, 4]
        regime_logits = self.regime_head(h)      # [B, 4]

        return {
            "event_logits": event_logits,
            "regime_logits": regime_logits,
            "embedding": h,
            "attn_weights": attn_weights,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        out = self.forward(x)
        event_prob = torch.sigmoid(out["event_logits"])
        regime_prob = torch.softmax(out["regime_logits"], dim=-1)
        return {
            **out,
            "event_prob": event_prob,
            "regime_prob": regime_prob,
        }


def build_model(input_dim: int) -> GRUEventModel:
    cfg = GRUEventConfig(input_dim=input_dim)
    return GRUEventModel(cfg)
