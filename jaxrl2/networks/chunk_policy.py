from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init

from jaxrl2.networks.attention_modules import CrossAttnTransformer
    
class ChunkPolicy(nn.Module):
    seq_len: int = 50
    action_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    action_magnitude: float = 1.0
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, cond_embed, training: bool = True):
        # 1. 用 CrossAttnTransformer 生成 sequence
        x = CrossAttnTransformer(
            seq_len=self.seq_len,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(cond_embed, training=training)  # (B, 50, hidden_dim)

        raw_values = nn.Dense(self.action_dim)(x)
        
        actions = jnp.tanh(raw_values) * self.action_magnitude
        
        return actions, raw_values
