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
        key: PRNGKey, critic: TrainState, target_critic: TrainState, res_actor: TrainState,
        batch: DatasetDict, discount: float, res_coeff: float, 
        backup_entropy: bool = False, critic_reduction: str = 'min', dp_unnorm_transform=None) -> Tuple[TrainState, Dict[str, float]]:
    observations = batch["observations"]
    actions = batch["actions"]
    next_observations = batch["next_observations"]
    rewards = batch['rewards']
    discounts = batch["discount"]
    masks = batch['masks']
    middle_actions = batch['middle_actions'] # (B, chunk, denoise_steps+1, action_dim)
    next_middle_actions = batch['next_middle_actions'] # (B, chunk, denoise_steps+1, action_dim)
    denoise_steps = middle_actions.shape[2] - 1
    batch_size = observations['pixels'].shape[0]
    
    # 2. Critic update (TD loss)
    # ---------------------
    def critic_loss_fn(params):
        critic_loss = 0.0
        critic_means = []
        for t in range(denoise_steps+1):
            current_times = jnp.full((batch_size, 1), t, dtype=jnp.int32)
            curr_actions = middle_actions[:, :, t, :]
            next_actions, _ = res_actor.apply_fn(
                {'params': res_actor.params}, next_observations, times=current_times
            ) 
            next_actions = next_actions[:, None, :] * res_coeff + next_middle_actions[:, :, t, :]
            curr_qs = critic.apply_fn({'params': params}, observations, curr_actions, current_times)
            tgt_qs = rewards + discounts * masks * target_critic.apply_fn(
                {'params': target_critic.params}, next_observations, next_actions, current_times)
            curr_td_err = curr_qs - tgt_qs
            critic_loss += jnp.mean(curr_td_err * curr_td_err)
            if t == 0 or t == denoise_steps-1 or t == denoise_steps:
                critic_means.append(curr_qs.mean())
        
        info = {
            "critic_loss": critic_loss,
            "q_0_mean": critic_means[0],
            "q_T_1_mean": critic_means[1],
            "q_T_mean": critic_means[2],
        }
        return critic_loss, info

    (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info


def update_clean_critic_wo_actor(
        key: PRNGKey, critic: TrainState, target_critic: TrainState, res_actor: TrainState,
        batch: DatasetDict, discount: float, res_coeff: float, 
        backup_entropy: bool = False, critic_reduction: str = 'min', dp_unnorm_transform=None) -> Tuple[TrainState, Dict[str, float]]:
    observations = batch["observations"]
    actions = batch["actions"]
    next_observations = batch["next_observations"]
    rewards = batch['rewards']
    discounts = batch["discount"]
    masks = batch['masks']
    middle_actions = batch['middle_actions'] # (B, chunk, denoise_steps+1, action_dim)
    next_middle_actions = batch['next_middle_actions'] # (B, chunk, denoise_steps+1, action_dim)
    denoise_steps = middle_actions.shape[2] - 1
    batch_size = observations['pixels'].shape[0]
    
    # 2. Critic update (TD loss)
    # ---------------------
    def critic_loss_fn(params):
        critic_loss = 0.0
        critic_means = []
        for t in range(denoise_steps+1):
            current_times = jnp.full((batch_size, 1), t, dtype=jnp.int32)
            curr_actions = middle_actions[:, :, t, :]
            next_actions = next_middle_actions[:, :, t, :]
            curr_qs = critic.apply_fn({'params': params}, observations, curr_actions, current_times)
            tgt_qs = rewards + discounts * masks * target_critic.apply_fn(
                {'params': target_critic.params}, next_observations, next_actions, current_times)
            curr_td_err = curr_qs - tgt_qs
            critic_loss += jnp.mean(curr_td_err * curr_td_err)
            if t == 0 or t == denoise_steps-1 or t == denoise_steps:
                critic_means.append(curr_qs.mean())
        
        info = {
            "critic_loss": critic_loss,
            "q_0_mean": critic_means[0],
            "q_T_1_mean": critic_means[1],
            "q_T_mean": critic_means[2],
        }
        return critic_loss, info

    (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
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