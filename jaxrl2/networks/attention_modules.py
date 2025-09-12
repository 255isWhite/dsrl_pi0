from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init



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

class QTransformer(nn.Module):
    seq_len: int = 50
    action_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, obs_embed, actions, training: bool = False):
        """
        obs_embed: (B, D_obs)      -> 编码好的观测 (256)
        actions:   (B, T, D_a)     -> 动作序列 (50, 32)
        return:    (B, 1)          -> Q 值
        """
        B, T, D_a = actions.shape

        # 投影动作到 hidden_dim
        x = nn.Dense(self.hidden_dim)(actions)   # (B, T, hidden_dim)

        # 加位置编码
        pos_emb = self.param(
            "pos_emb", nn.initializers.normal(stddev=0.02),
            (1, T, self.hidden_dim)
        )
        x = x + pos_emb

        # 堆叠 Cross-Attention Blocks
        for _ in range(self.num_layers):
            x = CrossAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, obs_embed, training=training)

        # 池化动作 token 表征
        x = x.mean(axis=1)   # (B, hidden_dim)

        return x
    
class QTransformer_nopool(nn.Module):
    seq_len: int = 50
    action_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, obs_embed, actions, training: bool = False):
        """
        obs_embed: (B, D_obs)      -> 编码好的观测 (256)
        actions:   (B, T, D_a)     -> 动作序列 (50, 32)
        return:    (B, 1)          -> Q 值
        """
        B, T, D_a = actions.shape

        # 投影动作到 hidden_dim
        x = nn.Dense(self.hidden_dim)(actions)   # (B, T, hidden_dim)

        # 加位置编码
        pos_emb = self.param(
            "pos_emb", nn.initializers.normal(stddev=0.02),
            (1, T, self.hidden_dim)
        )
        x = x + pos_emb

        # 堆叠 Cross-Attention Blocks
        for _ in range(self.num_layers):
            x = CrossAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(x, obs_embed, training=training)

        return x
    
    
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

        # # 投影到动作维度
        # actions = nn.Dense(self.action_dim)(x)  # (B, T, action_dim)
        return x