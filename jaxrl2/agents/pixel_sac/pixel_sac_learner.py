"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer, ActionChunkEncoder, ChunkPixelMultiplexer, LargePixelMultiplexer, TransformerPixelMultiplexer, BehaviorCloningGenerator, EnsembleQ
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_sac.actor_updater import update_actor, update_distill_actor, update_actor_with_bc
from jaxrl2.agents.pixel_sac.critic_updater import update_critic, update_critic_wo_actor
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.chunk_gaussian_policy import ChunkGaussianPolicy
from jaxrl2.networks.chunk_policy import ChunkPolicy
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.deterministic_policy import DeterministicPolicy
from jaxrl2.networks.values import StateActionEnsemble, StateValueEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any

def count_parameters(params):
    return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
def count_parameters_per_module(params):
    counts = {}
    def helper(subtree, path=()):
        if isinstance(subtree, dict):
            total = 0
            for k, v in subtree.items():
                sub_total = helper(v, path + (k,))
                total += sub_total
            # 如果这个 dict 有叶子参数，记录下来
            if total > 0:
                counts["/".join(path)] = total
            return total
        else:
            return np.prod(subtree.shape)
    helper(params)
    return counts

def count_top_modules(params):
    counts = {}
    for module, subparams in params.items():
        n = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(subparams))
        counts[module] = n
    return counts

@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter',  'aug_next', 'num_cameras'))
def _distill_jit(
    rng: PRNGKey, actor: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float,
    critic_reduction: str, color_jitter: bool, aug_next: bool, num_cameras: int, kl_coeff: float
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})
    
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_distill_actor(key, actor, batch, critic_reduction=critic_reduction, kl_coeff=kl_coeff)

    return rng, new_actor,{
        **actor_info,
    }
    
@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter',  'aug_next', 'num_cameras'))
def _update_bc_jit(
    rng: PRNGKey, actor: TrainState, distill_actor: TrainState, critic: TrainState,
    target_critic_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float,
    critic_reduction: str, color_jitter: bool, aug_next: bool, num_cameras: int, \
    kl_coeff: float, bc_coeff: float
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})
    
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(key, actor, critic, target_critic, temp, batch, discount, critic_reduction=critic_reduction)
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_with_bc(key, actor, distill_actor, new_critic, temp, batch, critic_reduction=critic_reduction, kl_coeff=kl_coeff, bc_coeff=bc_coeff)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    key, rng = jax.random.split(rng)
    new_distill_actor, distill_actor_info = update_distill_actor(key, distill_actor, batch, critic_reduction=critic_reduction, kl_coeff=kl_coeff)
    
    return rng, new_actor, new_distill_actor, new_critic, new_target_critic_params, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info,
    }, distill_actor_info

@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter',  'aug_next', 'num_cameras'))
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState,
    target_critic_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float,
    critic_reduction: str, color_jitter: bool, aug_next: bool, num_cameras: int, kl_coeff: float
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})
    
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic(key, actor, critic, target_critic, temp, batch, discount, critic_reduction=critic_reduction)
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch, critic_reduction=critic_reduction, kl_coeff=kl_coeff)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    return rng, new_actor, new_critic, new_target_critic_params, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }
    
@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter',  'aug_next', 'num_cameras'))
def _update_jit_wo_actor(
    rng: PRNGKey, critic: TrainState, actor: TrainState,
    target_critic_params: Params, batch: TrainState,
    discount: float, tau: float, target_entropy: float,
    critic_reduction: str, color_jitter: bool, aug_next: bool, num_cameras: int
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str,float]]:
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})
    
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    new_critic, critic_info = update_critic_wo_actor(key, critic, actor, target_critic, batch, discount, critic_reduction=critic_reduction)
    new_target_critic_params = soft_target_update(new_critic.params, target_critic_params, tau)
    

    return rng, new_critic, new_target_critic_params, {
        **critic_info,
    }


class PixelSACLearner(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 distill_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='group',
                 color_jitter = True,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=True,
                 use_bottleneck=True,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1,
                 kl_coeff: float = 1.0,
                 decay_kl: int = 0,
                 chunk_len: int = 20,
                 distill_action_dim: int = 7,
                 distill_hidden_dim: int = 256,
                 bc_coeff: float = 0.0,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.aug_next=aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.kl_coeff = kl_coeff
        self.decay_kl = decay_kl
        self.chunk_len = chunk_len
        self.distill_action_dim = distill_action_dim
        self.distill_hidden_dim = distill_hidden_dim
        self.bc_coeff = bc_coeff

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, distill_actor_key = jax.random.split(rng, 5)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])

        policy_def = ChunkGaussianPolicy(dropout_rate=dropout_rate, low=-action_magnitude, high=action_magnitude, hidden_dim=self.distill_hidden_dim)
        actor_def = LargePixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=self.distill_hidden_dim,
                                     use_bottleneck=use_bottleneck
                                     )
        print(actor_def)
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr),
                                  batch_stats=actor_batch_stats)

        # critic_def = StateValueEnsemble(hidden_dims, num_vs=num_qs)
        critic_def = EnsembleQ(
            encoder=encoder_def,
            latent_dim=self.distill_hidden_dim,
            use_bottleneck=use_bottleneck,
            num_qs=num_qs,
        )

        
        print(critic_def)
        critic_def_init = critic_def.init(critic_key, observations, actions)
        self._critic_init_params = critic_def_init['params']

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr),
                                   batch_stats=critic_batch_stats
                                   )
        target_critic_params = copy.deepcopy(critic_params)
        
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr),
                                 batch_stats=None)
        
        # residual head part
        distill_actor_def = BehaviorCloningGenerator(encoder=encoder_def,
                                     bc_action_dim=self.chunk_len*self.distill_action_dim,
                                     action_magnitude=action_magnitude,
                                     latent_dim=self.distill_hidden_dim,
                                     use_bottleneck=use_bottleneck,
                                     dropout_rate=dropout_rate
                                     )
        print(distill_actor_def)
        magic_actions = np.zeros((observations['pixels'].shape[0], 50, 32), dtype=np.float32)
        distill_actor_def_init = distill_actor_def.init(distill_actor_key, observations, magic_actions)
        distill_actor_params = distill_actor_def_init['params']
        distill_actor_batch_stats = distill_actor_def_init['batch_stats'] if 'batch_stats' in distill_actor_def_init else None

        distill_actor = TrainState.create(apply_fn=distill_actor_def.apply,
                                  params=distill_actor_params,
                                  tx=optax.adam(learning_rate=distill_lr),
                                  batch_stats=distill_actor_batch_stats)

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._distill_actor = distill_actor
        self._target_critic_params = target_critic_params
        self._temp = temp
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
        print(f'target_entropy: {self.target_entropy}')
        print(self.critic_reduction)
        # count params in how many Millions
        print('actor params (M): ', count_parameters(self._actor.params)/1e6)
        print('critic params (M): ', count_parameters(self._critic.params)/1e6)
        print('distill actor params (M): ', count_parameters(self._distill_actor.params)/1e6)
        print('total params (M): ', (count_parameters(self._actor.params)+count_parameters(self._critic.params) +count_parameters(self._distill_actor.params))/1e6)

        actor_counts = count_parameters_per_module(self._actor.params)
        print('actor params breakdown:')
        for module, n_params in actor_counts.items():
            print(f"{module:40s}: {n_params/1e6:.2f} M")
            
        critic_counts = count_parameters_per_module(self._critic.params)
        print('critic params breakdown:')
        for module, n_params in critic_counts.items():
            print(f"{module:40s}: {n_params/1e6:.2f} M")
            
        distill_actor_counts = count_parameters_per_module(self._distill_actor.params)
        print('distill actor params breakdown:')
        for module, n_params in distill_actor_counts.items():
            print(f"{module:40s}: {n_params/1e6:.2f} M")
            
        print('top-level actor params breakdown:')
        actor_top_counts = count_top_modules(self._actor.params)
        for module, n_params in actor_top_counts.items():
            print(f"{module:20s}: {n_params/1e6:.2f} M")
        print('top-level critic params breakdown:')
        critic_top_counts = count_top_modules(self._critic.params)
        for module, n_params in critic_top_counts.items():
            print(f"{module:20s}: {n_params/1e6:.2f} M")
        print('top-level distill actor params breakdown:')
        distill_actor_top_counts = count_top_modules(self._distill_actor.params)
        for module, n_params in distill_actor_top_counts.items():
            print(f"{module:20s}: {n_params/1e6:.2f} M")
        
        
    def update_with_bc(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, new_distill_actor, new_critic, new_target_critic, new_temp, info, distill_actor_info = _update_bc_jit(
            self._rng, self._actor, self._distill_actor, self._critic, self._target_critic_params, \
            self._temp, batch, self.discount, self.tau, self.target_entropy, self.critic_reduction, \
            self.color_jitter, self.aug_next, self.num_cameras, self.kl_coeff, self.bc_coeff
            )

        self._rng = new_rng
        self._actor = new_actor
        self._distill_actor = new_distill_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._temp = new_temp
        return info, distill_actor_info

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params, self._temp, batch, self.discount, self.tau, self.target_entropy, self.critic_reduction, self.color_jitter, self.aug_next, self.num_cameras, self.kl_coeff
            )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._temp = new_temp
        return info
    
    def update_wo_actor(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_critic, new_target_critic, info = _update_jit_wo_actor(
            self._rng, self._critic, self._actor, self._target_critic_params, batch, self.discount, self.tau, self.target_entropy, self.critic_reduction, self.color_jitter, self.aug_next, self.num_cameras
            )

        self._rng = new_rng
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        return info
    
    def distill(self, batch: FrozenDict) -> Dict[str, float]:
        new_rng, new_distill_actor, info = _distill_jit(
            self._rng, self._distill_actor, batch, self.discount, self.tau, self.target_entropy, self.critic_reduction, self.color_jitter, self.aug_next, self.num_cameras, self.kl_coeff
            )

        self._rng = new_rng
        self._distill_actor = new_distill_actor
        return info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        from examples.train_utils_sim import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs['rewards'])
        traj_images = []

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                next_obs_pixels = next_observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]
                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]

                q_value = get_value(action, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']
        print('restored from ', dir)
        
    
@functools.partial(jax.jit)
def get_value(action, observation, critic):
    input_collections = {'params': critic.params}
    q_pred = critic.apply_fn(input_collections, observation, action)
    return q_pred


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(q_estimates, rewards, masks, images):

    q_estimates_np = np.stack(q_estimates, 0).squeeze()
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates_np)])

    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    if len(q_estimates_np.shape) == 2:
        for i in range(q_estimates_np.shape[1]):
            axs[1].plot(q_estimates_np[:, i], linestyle='--', marker='o')
    else:
        axs[1].plot(q_estimates_np, linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])
    
    axs[3].plot(masks, linestyle='--', marker='d')
    axs[3].set_ylabel('masks')
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image