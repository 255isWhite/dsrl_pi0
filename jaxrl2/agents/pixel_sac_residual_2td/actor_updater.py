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
        tar_norm = actions.shape[-1]
        ac_norm = jnp.linalg.norm(actions, axis=-1)  # [batch]
        kl_loss = jnp.mean(0.5 * ac_norm**2 - (tar_norm - 1) * jnp.log(ac_norm + 1e-8))
        
        actor_loss = actor_loss + kl_loss * kl_coeff

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
            'mean_pi_batch_mean': mean_dist.mean(axis=0),
            'std_pi_batch_mean': std_diag_dist.mean(axis=0),
            # 'mean_pi_ac_mean': mean_dist.mean(axis=-1),
            # 'std_pi_ac_mean': std_diag_dist.mean(axis=-1),
            'kl_loss': kl_loss,
            'kl_coeff': kl_coeff,
            'ac_norm_avg': ac_norm.mean(),
            'ac_norm_max': ac_norm.max(),
            'ac_norm_min': ac_norm.min(),
            'ac_norm_std': ac_norm.std(),
            'tar_norm': jnp.sqrt(tar_norm-1),
        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info

def update_res_actor(key: PRNGKey, res_actor: TrainState, actor: TrainState, clean_critic: TrainState, 
                batch: DatasetDict, cross_norm:bool=False, critic_reduction:str='min', res_coeff:float=0.1, dp_unnorm_transform=None) -> Tuple[TrainState, Dict[str, float]]:
    key, key_act = jax.random.split(key, num=2)
    
    def res_actor_loss_fn(
            res_actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if hasattr(res_actor, 'batch_stats') and res_actor.batch_stats is not None:
            res_actions, raw_means, new_model_state = res_actor.apply_fn({'params': res_actor_params, 'batch_stats': res_actor.batch_stats}, batch['observations'], mutable=['batch_stats'])
        else: 
            res_actions, raw_means = res_actor.apply_fn({'params': res_actor_params}, batch['observations'])
            new_model_state = {}

        # actions = batch['norm_actions'].reshape(res_actions.shape[0],-1) + res_actions * res_coeff
        dist, means, log_stds = actor.apply_fn({'params': actor.params}, batch['observations'])
        noise, log_probs = dist.sample_and_log_prob(seed=key_act)
        
        # actions = dp_unnorm_transform({'actions': actions})['actions']

        if hasattr(clean_critic, 'batch_stats') and clean_critic.batch_stats is not None:
            qs, _ = clean_critic.apply_fn({'params': clean_critic.params, 'batch_stats': clean_critic.batch_stats}, batch['observations'],
                            res_actions, noise, mutable=['batch_stats'])
        else:    
            qs = clean_critic.apply_fn({'params': clean_critic.params}, batch['observations'], res_actions, noise)
        
        if critic_reduction == 'min':
            q = qs.min(axis=0)
        elif critic_reduction == 'mean':
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")
        
        actor_loss = - q.mean()
        
        things_to_log = {
            'actor_loss': actor_loss,
            'q_pi_in_actor': q.mean(),
            'q_pi_actor_all': q,
            'raw_means': raw_means.mean(),
            'raw_means_max': raw_means.max(),
            'raw_means_min': raw_means.min(),
            'raw_means_batch_mean': raw_means.mean(axis=0),
            'raw_means_ac_mean': raw_means.mean(axis=-1),
        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(res_actor_loss_fn, has_aux=True)(res_actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = res_actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = res_actor.apply_gradients(grads=grads)

    return new_actor, info