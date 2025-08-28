from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import flax.nnx as nnx
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey
from jaxrl2.networks.learned_std_normal_policy import TanhMultivariateNormalDiag


def update_critic_wo_inference(
        key: PRNGKey, critic: TrainState, actor: TrainState,
        target_critic: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min',
) -> Tuple[TrainState, Dict[str, float]]:   
    next_clean_actions = batch['next_actions'].reshape(batch['next_actions'].shape[0],-1)

    # 2. target critic
    next_qs = target_critic.apply_fn({'params': target_critic.params}, batch['next_observations'], next_clean_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplementedError()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q


    middle_actions = batch['middle_actions']
    
    # 3. rng
    rng, subkey = jax.random.split(key)

    # 4. critic loss fn
    def critic_loss_fn(critic_params: Params):
        # forward critic Q
        qs = critic.apply_fn({'params': critic_params},
                            batch['observations'], batch['actions'])

        # critic TD loss
        td_err = qs - target_q
        critic_loss = jnp.mean(td_err * td_err)   # 避免 (..)**2.mean() 这种两步写法

        mid_loss = 0.0        
        guidance_clean_actions = middle_actions[:, :, 0, :]
        guidance_tgt_qs = jax.lax.stop_gradient(
            critic.apply_fn({'params': critic_params}, batch['observations'], guidance_clean_actions)
        )
        
        denoise_steps = middle_actions.shape[2] - 1
        for t in range(1, denoise_steps + 1):
            curr_actions = middle_actions[:, :, t, :]
            current_time = jnp.full((curr_actions.shape[0], 1), t, dtype=jnp.int32)
            curr_qs = critic.apply_fn({'params': critic_params}, batch['observations'], curr_actions, current_time)
            curr_td_err = curr_qs - guidance_tgt_qs
            mid_loss += jnp.mean(curr_td_err * curr_td_err)

        total_loss = critic_loss + mid_loss
        
        return total_loss, {
            'critic_loss': critic_loss,
            'mid_loss': mid_loss,
            'total_loss': total_loss,
            'q': qs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'guidance_tgt_q': guidance_tgt_qs.mean(),
        }

    # 5. grad + update
    grad_fn = jax.grad(critic_loss_fn, has_aux=True)
    grads, info = grad_fn(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info


def update_critic_with_inference(
        key: PRNGKey, critic: TrainState, actor: TrainState,
        target_critic: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min', guidance_scale: float = 1.0,
        pi0_params=None, pi0_def=None, task_prompt=None
) -> Tuple[TrainState, Dict[str, float]]:   

    # 1. actor sample
    rng, subkey = jax.random.split(key)
    
    # === 调用 Pi0.sample_guidance ===
    pi0_model = nnx.merge(pi0_def, pi0_params)   # 合并成实例
    action_horizon = pi0_model.action_horizon
    action_dim = pi0_model.action_dim
    noise=jax.random.normal(
        subkey,
        (batch['actions'].shape[0], action_horizon, action_dim),
    )

    # === Pi0 guidance ===
    next_clean_actions = pi0_model.sample_actions_train(
        {**batch["next_pi_zero_obs"], "prompt": task_prompt},
        noise=noise,   # 预先生成，避免 trace 时插入 rng op
        guidance={"apply_fn": target_critic.apply_fn, "params": critic.params},
        guidance_obs=batch['observations'],
        guidance_scale=guidance_scale,
    )

    # 2. target critic
    next_qs = target_critic.apply_fn({'params': target_critic.params}, batch['next_observations'], next_clean_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplementedError()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q


    middle_actions = batch['middle_actions']
    
    # 3. rng
    rng, subkey = jax.random.split(key)

    # 4. critic loss fn
    def critic_loss_fn(critic_params: Params):
        # forward critic Q
        qs = critic.apply_fn({'params': critic_params},
                            batch['observations'], batch['actions'])

        # critic TD loss
        td_err = qs - target_q
        critic_loss = jnp.mean(td_err * td_err)   # 避免 (..)**2.mean() 这种两步写法

        mid_loss = 0.0        
        guidance_clean_actions = middle_actions[:, :, 0, :]
        guidance_tgt_qs = jax.lax.stop_gradient(
            critic.apply_fn({'params': critic_params}, batch['observations'], guidance_clean_actions)
        )
        
        denoise_steps = 10
        for t in range(1, denoise_steps + 1):
            curr_actions = middle_actions[:, :, t, :]
            current_time = jnp.full((curr_actions.shape[0], 1), t, dtype=jnp.int32)
            curr_qs = critic.apply_fn({'params': critic_params}, batch['observations'], curr_actions, current_time)
            curr_td_err = curr_qs - guidance_tgt_qs
            mid_loss += jnp.mean(curr_td_err * curr_td_err)

        total_loss = critic_loss + mid_loss
        
        return total_loss, {
            'critic_loss': critic_loss,
            'mid_loss': mid_loss,
            'total_loss': total_loss,
            'q': qs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'guidance_tgt_q': guidance_tgt_qs.mean(),
        }

    # 5. grad + update
    grad_fn = jax.grad(critic_loss_fn, has_aux=True)
    grads, info = grad_fn(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info



# def update_critic_wo_actor(
#         key: PRNGKey, critic: TrainState, actor: TrainState,
#         target_critic: TrainState, batch: DatasetDict,
#         discount: float, backup_entropy: bool = False,
#         critic_reduction: str = 'min',
#         pi0_params=None, pi0_def=None, task_prompt=None) -> Tuple[TrainState, Dict[str, float]]:   
#     demo_dist, means, log_stds = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    
    
#     norm_dist = make_std_gaussian_like(demo_dist)
#     next_actions, next_log_probs = norm_dist.sample_and_log_prob(seed=key)
#     next_qs = target_critic.apply_fn({'params': target_critic.params},
#                                     batch['next_observations'], next_actions)
#     if critic_reduction == 'min':
#         next_q = next_qs.min(axis=0)
#     elif critic_reduction == 'mean':
#         next_q = next_qs.mean(axis=0)
#     else:
#         raise NotImplemented()

#     target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q
    
#     pi_zero_obs = {**batch["pi_zero_obs"], "prompt": task_prompt}
#     rng, subkey = jax.random.split(key)
    
#     def critic_loss_fn(
#             critic_params: Params, rng_key) -> Tuple[jnp.ndarray, Dict[str, float]]:
#         qs = critic.apply_fn({'params': critic_params}, batch['observations'],
#                             batch['actions'])
#         critic_loss = ((qs - target_q)**2).mean()
        
#         # 用新的 critic params 重新 forward，得到所有 q_sums
#         flow_q_sums = pi0_def.sample_guidance(
#             pi_zero_obs, batch['observations'], rng_key, critic_params
#         )

#         # target Q
#         guidance_tgt_q = jax.lax.stop_gradient(flow_q_sums[0])  # (B,)
#         mid_q_sums = flow_q_sums[1:]    # (n, B)  

#         # 随机采样一部分 step
#         # idx = jax.random.choice(rng_key, mid_q_sums.shape[1], (1,), replace=False)
#         # sampled = mid_q_sums[:, idx]
#         # guidance_tgt_q = guidance_tgt_q[idx]
#         # mid_losses = ((sampled - guidance_tgt_q[None, :]) ** 2).mean()
        
#         # 所有 mid loss
#         mid_losses = ((mid_q_sums - guidance_tgt_q[None, :]) ** 2).mean(axis=1)
        
#         critic_loss = critic_loss + mid_losses.mean()
        
#         return critic_loss, {
#             'critic_loss': critic_loss,
#             'q': qs.mean(),
#             'target_actor_entropy': -next_log_probs.mean(),
#             'next_actions_sampled': next_actions.mean(),
#             'next_log_probs': next_log_probs.mean(),
#             'next_q_pi': next_qs.mean(),
#             'target_q': target_q.mean(),
#             'next_actions_mean': next_actions.mean(),
#             'next_actions_std': next_actions.std(),
#             'next_actions_min': next_actions.min(),
#             'next_actions_max': next_actions.max(),
#             'next_log_probs': next_log_probs.mean(),
#             'q_batch_mean': qs.mean(axis=0),
#         }

#     # 打印计算图
#     grad_fn = jax.grad(critic_loss_fn, has_aux=True)
#     # print(jax.make_jaxpr(grad_fn)(critic.params, subkey))
#     grads, info = grad_fn(critic.params, subkey)
#     new_critic = critic.apply_gradients(grads=grads)


#     return new_critic, info

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