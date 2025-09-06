from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey
from jaxrl2.networks.learned_std_normal_policy import TanhMultivariateNormalDiag

def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min', td3_noise_scale: float = 0.2,
        action_magnitude: float = 3.0) -> Tuple[TrainState, Dict[str, float]]:
    next_actions, next_raw_means = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    key, subkey = jax.random.split(key)
    td3_noise = jnp.clip(
        jax.random.normal(subkey, shape=next_actions.shape) * td3_noise_scale, -0.5, 0.5
    )
    next_actions = jnp.clip(next_actions + td3_noise, -action_magnitude, action_magnitude)
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['actions'])
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'q_batch_mean': qs.mean(axis=0),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info


def update_critic_wo_actor(
        key: PRNGKey, critic: TrainState, actor: TrainState,
        target_critic: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min') -> Tuple[TrainState, Dict[str, float]]:   

    demo_actions, _ = actor.apply_fn({'params': actor.params}, batch['next_observations'])

    # generate a random action from Gaussian(0,1) with the same shape as demo_actions
    key, subkey = jax.random.split(key)
    random_actions = jax.random.normal(subkey, shape=demo_actions.shape)
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], random_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['actions'])
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'q_batch_mean': qs.mean(axis=0),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info

def make_std_gaussian_like(dist):
    # 从已有 dist 拿 shape
    loc_shape = dist.distribution.loc.shape  # e.g. (batch_size, action_dim)
    zeros = jnp.zeros(loc_shape)
    ones = jnp.ones(loc_shape)

    # 保证没有梯度
    zeros = jax.lax.stop_gradient(zeros)
    ones = jax.lax.stop_gradient(ones)

    # 保持 low/high 一致（如果有）
    low = dist.low
    high = dist.high

    return TanhMultivariateNormalDiag(loc=zeros, scale_diag=ones, low=low, high=high)