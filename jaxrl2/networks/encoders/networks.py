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


def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]):
    if hasattr(x, 'values'):
        obs = []
        for k, v in sorted(x.items()):
            # if k == "actions":
            #     v = v[:, 0:1, ...]
            if k == 'state': # flatten action chunk to 1D
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                # v = jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])])
            elif k == 'prev_action' or k == 'actions' or k == 'noise':
                if v.ndim > 2:
                    # deal with action chunk
                    obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                else:
                    obs.append(v)
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1)
    else:
        return x
    


class ActionChunkEncoder(nn.Module):
    hidden_dim: int = 128
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
    chunk_encoder: nn.Module=None

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 noise: jnp.ndarray,
                 training: bool = False):
        observations = FrozenDict(observations)
        
        # # print all shapes
        # for k,v in observations.items():
        #     print(f"obs key: {k}, shape: {v.shape}")
        # print(f"action shape: {noise.shape}")

        x = self.encoder(observations['pixels'], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim * 2, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        x = observations.copy(add_or_replace={'pixels': x})

        y = observations['state']
        y = nn.Dense(self.latent_dim, kernel_init=xavier_init())(y)
        y = nn.LayerNorm()(y)
        y = nn.tanh(y)
        x = x.copy(add_or_replace={'state': y})
        
        noise = self.chunk_encoder(noise, train=training)
        x = x.copy(add_or_replace={'noise': noise})
        
        return self.network(x, training=training)

import flax.linen as nn

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
    
    
class CrossAttentionBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, cond, training: bool = False):
        """
        x: (B, T, hidden_dim)  -> 动作序列 token
        cond: (B, D_cond)      -> 图像/状态条件 latent
        """
        # 把 cond 变成序列 token (B, 1, hidden_dim)
        cond = nn.Dense(self.hidden_dim)(cond)[:, None, :]

        # LayerNorm + CrossAttention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=not training
        )(x, cond)   # Q = x, K/V = cond
        x = residual + x

        # FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        return residual + x


class CrossAttnTransformer(nn.Module):
    seq_len: int = 50
    action_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, cond_embed, training: bool = False):
        """
        cond_embed: (B, D_cond)  已经是图像+state latent
        返回: (B, seq_len, action_dim)
        """
        B, D = cond_embed.shape

        # 初始化动作序列 token (learned query)
        action_tokens = self.param(
            "action_tokens", nn.initializers.normal(stddev=0.02),
            (1, self.seq_len, self.hidden_dim)
        )
        x = jnp.tile(action_tokens, (B, 1, 1))  # (B, T, hidden_dim)

        # 堆叠 Cross-Attention Block
        for _ in range(self.num_layers):
            x = CrossAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, cond_embed, training=training)

        # 投影到动作维度
        actions = nn.Dense(self.action_dim)(x)  # (B, T, action_dim)
        return actions