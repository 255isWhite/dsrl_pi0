from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init

class DeterministicPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    action_magnitude: float = 1.0

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False,
                 ) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        raw_means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        actions = jnp.tanh(raw_means) * self.action_magnitude
        
        return actions, raw_means