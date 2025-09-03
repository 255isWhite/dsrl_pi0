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
        critic_reduction: str = 'min') -> Tuple[TrainState, Dict[str, float]]:
    dist, means, log_stds = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    if backup_entropy:
        target_q -= batch["discount"] * batch['masks'] * temp.apply_fn(
            {'params': temp.params}) * next_log_probs

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['actions'])
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs.mean(),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_log_probs': next_log_probs.mean(),
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
    demo_dist, means, log_stds = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    
    norm_dist = make_std_gaussian_like(demo_dist)
    next_actions, next_log_probs = norm_dist.sample_and_log_prob(seed=key)
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
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_log_probs': next_log_probs.mean(),
            'q_batch_mean': qs.mean(axis=0),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info


def update_clean_critic(
        key: PRNGKey, clean_critic: TrainState, target_clean_critic: TrainState, res_actor: TrainState,
        batch: DatasetDict, discount: float, res_coeff: float, res_prob: float, 
        backup_entropy: bool = False, critic_reduction: str = 'min', dp_unnorm_transform=None) -> Tuple[TrainState, Dict[str, float]]:
    
    res_actions, raw_means = res_actor.apply_fn({'params': res_actor.params}, batch['observations'])
    next_res_actions, next_raw_means = res_actor.apply_fn({'params': res_actor.params}, batch['next_observations'])
    next_norm_actions = batch['next_norm_actions'].reshape(next_res_actions.shape[0], -1)
    next_qs = target_clean_critic.apply_fn({'params': target_clean_critic.params},
                                     batch['next_observations'], next_res_actions + next_norm_actions)
    
    # print(f"shape of next_res_actions: {next_res_actions.shape}, shape of next_norm_actions: {next_norm_actions.shape}")
    
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    def clean_critic_loss_fn(
            clean_critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        
        # rng, subkey = jax.random.split(key)
        # rand_val = jax.random.uniform(subkey, ())  # 标量 in [0,1)
        # clean_actions = jax.lax.cond(
        #     rand_val < res_prob,                                # 谓词是 tracer 也没问题
        #     lambda _: batch['norm_actions'].reshape(res_actions.shape[0],-1) + res_actions * res_coeff,  # true 分支
        #     lambda _: batch['norm_actions'].reshape(res_actions.shape[0],-1),                   # false 分支
        #     operand=None
        # )
        clean_actions = batch['actual_norm_actions'].reshape(res_actions.shape[0],-1)
        # clean_actions = dp_unnorm_transform({'actions': clean_actions})['actions']
                
        clean_qs = clean_critic.apply_fn({'params': clean_critic_params}, batch['observations'],
                             clean_actions)
        
        clean_critic_loss = ((clean_qs - target_q)**2).mean()
        return clean_critic_loss, {
            'clean_critic_loss': clean_critic_loss,
            'q': clean_qs.mean(),
            'q_batch_mean': clean_qs.mean(axis=0),
            'bacth_norm_actions_mean': batch['norm_actions'].mean(axis=0),
            'bacth_norm_actions_std': batch['norm_actions'].std(axis=0),
            'bacth_norm_actual_actions_mean': batch['actual_norm_actions'].mean(axis=0),
            'bacth_norm_actual_actions_std': batch['actual_norm_actions'].std(axis=0),
            'res_actions_mean': res_actions.mean(axis=0),
            'res_actions_std': res_actions.std(axis=0),
            'res_actions_min': res_actions.min(axis=0),
            'res_actions_max': res_actions.max(axis=0),
        }

    grads, info = jax.grad(clean_critic_loss_fn, has_aux=True)(clean_critic.params)
    new_critic = clean_critic.apply_gradients(grads=grads)

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