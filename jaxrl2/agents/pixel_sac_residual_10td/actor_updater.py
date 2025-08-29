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
            if cross_norm:
                next_dist, means, log_stds, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'], mutable=['batch_stats'])
            else:
                next_dist, means, log_stds = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'])
        else:
            dist, means, log_stds = actor.apply_fn({'params': actor_params}, batch['observations'])
            next_dist, means, log_stds = actor.apply_fn({'params': actor_params}, batch['next_observations'])
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

def update_res_actor(key: PRNGKey, actor: TrainState, critic: TrainState, 
                batch: DatasetDict, cross_norm:bool=False, critic_reduction:str='min', \
                res_coeff:float=0.1, dp_unnorm_transform=None) -> Tuple[TrainState, Dict[str, float]]:
    observations = batch["observations"]
    middle_actions = batch['middle_actions'] # (B, chunk, denoise_steps+1, action_dim)
    denoise_steps = middle_actions.shape[2] - 1
    batch_size = observations['pixels'].shape[0]
    
    def actor_loss_fn(
            params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        actor_loss = 0.0
        actor_q_means = []
        raw_value_list = []
        for t in range(denoise_steps):
            current_times = jnp.full((batch_size, 1), t, dtype=jnp.int32)
            curr_actions, raw_values = actor.apply_fn(
                {'params': params}, observations, times=current_times
            )
            curr_actions = curr_actions[:, None, :] * res_coeff + middle_actions[:, :, t, :]

            curr_qs = critic.apply_fn({'params': critic.params}, observations, curr_actions, current_times)
            
            actor_loss -= curr_qs.mean()
            if t == 0 or t == denoise_steps-1:
                actor_q_means.append(curr_qs.mean())
            raw_value_list.append(raw_values)

        raw_value_array = jnp.concatenate(raw_value_list, axis=-1) # (B, denoise_steps * action_dim)
        info = {
            "actor_loss": actor_loss,
            "actor_q_0_mean": actor_q_means[0],
            "actor_q_T_1_mean": actor_q_means[1],
            "actor_raw_max": raw_value_array.max(),
            "actor_raw_min": raw_value_array.min(),
            "actor_raw_mean": raw_value_array.mean(),
            "actor_raw_std": raw_value_array.std()
        }
        return actor_loss, info

    (loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info