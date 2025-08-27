from typing import Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import distrax

ModuleDef = Any

class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))
    

class PixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool=True
    denoise_steps: int=10
    time_dim: int=8

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 times: Optional[jnp.ndarray] = None,
                 training: bool = False):
        observations = FrozenDict(observations)

        x = self.encoder(observations['pixels'], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        obs = observations.copy(add_or_replace={'pixels': x})
            
        # ---- Add time embedding ----
        # TBC
        if times is None:
            # 从 pixels 里推 batch size
            B = observations['pixels'].shape[0]
            times = jnp.zeros((B, 1), dtype=jnp.int32)  # shape [B, 1]
        
        # embedding lookup
        t_emb = nn.Embed(num_embeddings=self.denoise_steps+1, features=self.time_dim)(
            times.astype(jnp.int32)
        )
        obs = obs.copy(add_or_replace={'time': t_emb})

        # # print key and shape of obs
        # for k, v in obs.items():
        #     print(f"key: {k}, shape: {v.shape}")
        # # action
        # if actions is not None:
        #     print(f"action shape: {actions.shape}")

        # print('fully connected keys', x.keys())
        if actions is None:
            return self.network(obs, training=training)
        else:
            return self.network(obs, actions, training=training)
