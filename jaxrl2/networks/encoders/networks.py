from typing import Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import numpy as np
from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple
import distrax
from jaxrl2.networks.values.attention_modules import CrossAttnTransformer


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

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        observations = FrozenDict(observations)

        x = self.encoder(observations['pixels'], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        x = observations.copy(add_or_replace={'pixels': x})

        # print('fully connected keys', x.keys())
        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)


class LargePixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool=True

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 training: bool = False):
        observations = FrozenDict(observations)
        
        # # print _shape
        # for k, v in observations.items():
        #     print (k, v.shape)
        # if actions is not None:
        #     print ('action shape: ', actions.shape)
        # if noise is not None:
        #     print ('noise shape: ', noise.shape)

        x = self.encoder(observations['pixels'], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        print(f"post-encoder flattened shape: {x.shape}")

        y = observations['state'].reshape(observations['state'].shape[0], -1)
        # y = nn.Dense(self.latent_dim, kernel_init=xavier_init())(y)
        # y = nn.LayerNorm()(y)
        # y = nn.tanh(y)
        print(f"post-state flattened shape: {y.shape}")

        x = jnp.concatenate([x, y], axis=-1)        

        print(f"final flattened shape: {x.shape}")
        
        return self.network(x, training=training)
        

class ActionChunkEncoder(nn.Module):
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    out_dim: int = 32
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x shape: (B, 50, 32)
        if x.ndim == 2:  # (B, 1600)
            B, D = x.shape
            assert D == 50 * 32, f"Expected 1600 dims, got {D}"
            x = x.reshape(B, 50, 32)
            B, T, D = x.shape
        elif x.ndim == 3:  # (B, 50, 32)
            B, T, D = x.shape
            assert T == 50 and D == 32, f"Expected (B,50,32), got (B,{T},{D})"
        else:
            raise ValueError(f"Unexpected x shape {x.shape}, expect (B,50,32) or (B,1600)")

        # 先线性投影到 hidden_dim
        x = nn.Dense(self.hidden_dim)(x)   # (B, 50, hidden_dim)

        # 加位置编码（可学习）
        pos_emb = self.param("pos_emb", nn.initializers.normal(stddev=0.02), (1, T, self.hidden_dim))
        x = x + pos_emb

        # Transformer Encoder 堆叠
        for _ in range(self.num_layers):
            x = nn.SelfAttention(num_heads=self.num_heads)(x)
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)

        # 聚合序列 → 得到全局特征
        x = x.reshape(B, -1)

        # 输出 (比如再映射回 32 维 action embedding)
        out = nn.Dense(self.out_dim)(x)     # (B, 32)
        return out
    

class ChunkPixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool=True
    seq_len: int = 50
    action_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray],
                 training: bool = False):
        observations = FrozenDict(observations)
        
        # print all shapes
        for k,v in observations.items():
            print(f"obs key: {k}, shape: {v.shape}")
        print(f"Q action shape: {actions.shape}")
        print(f"Q pixel shape: {observations['pixels'].shape}")
        x = self.encoder(observations['pixels'], training)
        print(f"Q post-encoder flattened shape: {x.shape}")
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        y = observations['state'].reshape(observations['state'].shape[0], -1)
        
        x = jnp.concatenate([x, y], axis=-1)    
        print(f"Q combined pixel+state shape: {x.shape}")
        
        cross_features = CrossAttnTransformer(
            seq_len=self.seq_len,
            action_dim=self.action_dim,
            hidden_dim=self.latent_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(x, training=training)  # (B, 50, hidden_dim)
        
        cross_features = cross_features.reshape(cross_features.shape[0], -1)  # (B, 50*hidden_dim)
        cross_features = nn.Dense(self.latent_dim * 2, kernel_init=xavier_init())(cross_features)
        
        return self.network(cross_features, training=training)