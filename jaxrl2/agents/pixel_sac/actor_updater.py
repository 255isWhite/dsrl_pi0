from audioop import cross
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_actor(key: PRNGKey, actor: TrainState, critic: TrainState,
                 temp: TrainState, batch: DatasetDict, cross_norm:bool=False, critic_reduction:str='min', kl_coeff:float=1.0) -> Tuple[TrainState, Dict[str, float]]:

    key, key_act = jax.random.split(key, num=2)

    def actor_loss_fn(
            actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['observations'], mutable=['batch_stats'])
        else:
            dist, means, log_stds = actor.apply_fn({'params': actor_params}, batch['observations'])
            new_model_state = {}
        
        # For logging only
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        actions, log_probs = dist.sample_and_log_prob(seed=key_act)

        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            qs, _ = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats}, batch['observations'],
                            actions, mutable=['batch_stats'])
        else:    
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
        
        if critic_reduction == 'min':
            q = qs.min(axis=0)
        elif critic_reduction == 'mean':
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")
        actor_loss = (log_probs * temp.apply_fn({'params': temp.params}) - q).mean()
        
        # a. KL only MU
        # kl_loss = jnp.mean(jnp.square(means))
        
        # b. KL MU and STD
        # stds = jnp.exp(log_stds)
        # kl_loss = 0.5 * jnp.mean(stds**2 + means**2 - 1.0 - 2.0 * log_stds)
        
        # c. KL Gaussian
        tar_norm = 32
        ac_norm = jnp.linalg.norm(actions[...,:32], axis=-1)  # [batch]
        kl_loss = jnp.mean(0.5 * ac_norm**2 - (tar_norm - 1) * jnp.log(ac_norm + 1e-8))

        actor_loss = actor_loss + kl_loss * kl_coeff

        things_to_log = {
            'actor_loss': actor_loss,
            'actor_entropy': -log_probs.mean(),
            'actor_q_pi_in_actor': q.mean(),
            
            'actor_sac_mean_pi_norm': mean_dist_norm.mean(),
            'actor_sac_std_pi_norm': std_dist_norm.mean(),
            'actor_sac_mean_pi_avg': mean_dist.mean(),
            'actor_sac_mean_pi_max': mean_dist.max(),
            'actor_sac_mean_pi_min': mean_dist.min(),
            'actor_sac_std_pi_avg': std_diag_dist.mean(),
            'actor_sac_std_pi_max': std_diag_dist.max(),
            'actor_sac_std_pi_min': std_diag_dist.min(),
            # 'mean_pi_batch_mean': mean_dist.mean(axis=0),
            # 'std_pi_batch_mean': std_diag_dist.mean(axis=0),
            # 'mean_pi_ac_mean': mean_dist.mean(axis=-1),
            # 'std_pi_ac_mean': std_diag_dist.mean(axis=-1),
            'actor_kl_loss': kl_loss,
            'actor_ac_norm_avg': ac_norm.mean(),
            'actor_ac_norm_max': ac_norm.max(),
            'actor_ac_norm_min': ac_norm.min(),
            'actor_ac_norm_std': ac_norm.std(),
            'actor_tar_norm': jnp.sqrt(tar_norm-1),

            'actor_noise_mean_pi_avg': mean_dist[..., :32].mean(),
            'actor_noise_mean_std_avg': std_diag_dist[..., :32].mean(),

            'actor_noise_mean': actions[...,:32].mean(),
            'actor_noise_std': actions[...,:32].std(),
            'actor_noise_min': actions[...,:32].min(),
            'actor_noise_max': actions[...,:32].max(),

            'actor_residual_mean': actions[...,32:].mean(),
            'actor_residual_std': actions[...,32:].std(),
            'actor_residual_min': actions[...,32:].min(),
            'actor_residual_max': actions[...,32:].max(),

        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info