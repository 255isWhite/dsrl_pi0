from typing import Union
from typing import Iterable, Optional
import jax 
import gym
import gym.spaces
import numpy as np
import pickle

import copy

from jaxrl2.data.dataset import Dataset, DatasetDict
import collections
from flax.core import frozen_dict

def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int, chunk_length: int = 1, res_action_dim: int = 7):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity
        self.chunk_length = chunk_length
        self.magic_shape = (chunk_length, res_action_dim)

        print("making replay buffer of capacity ", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        next_observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        norm_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
        next_norm_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
        clean_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
        actual_norm_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
        next_actual_norm_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
        distill_noise_actions = np.empty((self.capacity, 50, 32), dtype=self.action_space.dtype)
        distill_clean_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
        rewards = np.empty((self.capacity, ), dtype=np.float32)
        masks = np.empty((self.capacity, ), dtype=np.float32)
        discount = np.empty((self.capacity, ), dtype=np.float32)

        self.data = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': actions,
            'next_actions': next_actions,
            'norm_actions': norm_actions,
            'next_norm_actions': next_norm_actions,
            'clean_actions': clean_actions,
            'actual_norm_actions': actual_norm_actions,
            'next_actual_norm_actions': next_actual_norm_actions,
            'rewards': rewards,
            'masks': masks,
            'discount': discount,
            'distill_noise_actions': distill_noise_actions,
            'distill_clean_actions': distill_clean_actions,
        }

        self.size = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_bounds = dict()
        self.streaming_buffer_size = None # this is for streaming the online data

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    def increment_traj_counter(self):
        self.traj_bounds[self._traj_counter] = (self._start, self.size) # [start, end)
        self._start = self.size
        self._traj_counter += 1

    def get_random_trajs(self, num_trajs: int):
        self.which_trajs = np.random.randint(0, self._traj_counter, num_trajs)
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []
        discount_list = []
        norm_actions_list = []
        next_norm_actions_list = []
        clean_actions_list = []
        actual_norm_actions_list = []
        next_actual_norm_actions_list = []
        distill_noise_actions_list = []
        distill_clean_actions_list = []

        for i in self.which_trajs:
            start, end = self.traj_bounds[i]
            
            # handle this as a dictionary
            obs_dict_curr_traj = dict()
            for k in self.data['observations']:
                obs_dict_curr_traj[k] = self.data['observations'][k][start:end]
            observations_list.append(obs_dict_curr_traj)
            
            next_obs_dict_curr_traj = dict()
            for k in self.data['next_observations']:
                next_obs_dict_curr_traj[k] = self.data['next_observations'][k][start:end]    
            next_observations_list.append(next_obs_dict_curr_traj)
            
            actions_list.append(self.data['actions'][start:end])
            norm_actions_list.append(self.data['norm_actions'][start:end])
            next_norm_actions_list.append(self.data['next_norm_actions'][start:end])
            clean_actions_list.append(self.data['clean_actions'][start:end])
            actual_norm_actions_list.append(self.data['actual_norm_actions'][start:end])
            next_actual_norm_actions_list.append(self.data['next_actual_norm_actions'][start:end])
            rewards_list.append(self.data['rewards'][start:end])
            terminals_list.append(1-self.data['masks'][start:end])
            masks_list.append(self.data['masks'][start:end])
            distill_noise_actions_list.append(self.data['distill_noise_actions'][start:end])
            distill_clean_actions_list.append(self.data['distill_clean_actions'][start:end])


        
        batch = {
            'observations': observations_list,
            'next_observations': next_observations_list,
            'actions': actions_list,
            'norm_actions': norm_actions_list,
            'next_norm_actions': next_norm_actions_list,
            'clean_actions': clean_actions_list,
            'actual_norm_actions': actual_norm_actions_list,
            'next_actual_norm_actions': next_actual_norm_actions_list,
            'rewards': rewards_list,
            'terminals': terminals_list,
            'masks': masks_list,
            'distill_noise_actions': distill_noise_actions_list,
            'distill_clean_actions': distill_clean_actions_list,
        }
        return batch
        
    def insert(self, data_dict: DatasetDict):
        if self.size == self.capacity:
            # Double the capacity
            observations = _init_replay_dict(self.observation_space, self.capacity)
            next_observations = _init_replay_dict(self.observation_space, self.capacity)
            actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            norm_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
            next_norm_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
            clean_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
            actual_norm_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
            next_actual_norm_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)
            next_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            rewards = np.empty((self.capacity, ), dtype=np.float32)
            masks = np.empty((self.capacity, ), dtype=np.float32)
            discount = np.empty((self.capacity, ), dtype=np.float32)
            distill_noise_actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
            distill_clean_actions = np.empty((self.capacity, *self.magic_shape), dtype=self.action_space.dtype)

            data_new = {
                'observations': observations,
                'next_observations': next_observations,
                'actions': actions,
                'norm_actions': norm_actions,
                'next_norm_actions': next_norm_actions,
                'clean_actions': clean_actions,
                'actual_norm_actions': actual_norm_actions,
                'next_actual_norm_actions': next_actual_norm_actions,
                'next_actions': next_actions,
                'rewards': rewards,
                'masks': masks,
                'discount': discount,
                'distill_noise_actions': distill_noise_actions,
                'distill_clean_actions': distill_clean_actions,
            }

            for x in data_new:
                if isinstance(self.data[x], np.ndarray):
                    self.data[x] = np.concatenate((self.data[x], data_new[x]), axis=0)
                elif isinstance(self.data[x], dict):
                    for y in self.data[x]:
                        self.data[x][y] = np.concatenate((self.data[x][y], data_new[x][y]), axis=0)
                else:
                    raise TypeError()
            self.capacity *= 2


        for x in data_dict:
            if x in self.data:
                if isinstance(data_dict[x], dict):
                    for y in data_dict[x]:
                        self.data[x][y][self.size] = data_dict[x][y]
                else:                        
                    self.data[x][self.size] = data_dict[x]
        self.size += 1
    
    def compute_action_stats(self):
        actions = self.data['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.data['actions'] = (self.data['actions'] - action_stats['mean']) / action_stats['std']
        self.data['next_actions'] = (self.data['next_actions'] - action_stats['mean']) / action_stats['std']

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if self.streaming_buffer_size:
            indices = np.random.randint(0, self.streaming_buffer_size, batch_size)
        else:
            indices = np.random.randint(0, self.size, batch_size)
        data_dict = {}
        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                data_dict[x] = self.data[x][indices]
            elif isinstance(self.data[x], dict):
                data_dict[x] = {}
                for y in self.data[x]:
                    data_dict[x][y] = self.data[x][y][indices]
            else:
                raise TypeError()
        
        return frozen_dict.freeze(data_dict)

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


    def save(self, filename):
        save_dict = dict(
            data=self.data,
            size = self.size,
            _traj_counter = self._traj_counter,
            _start=self._start,
            traj_bounds=self.traj_bounds
        )
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f, protocol=4)


    def restore(self, filename):
        with open(filename, 'rb') as f:
            save_dict = pickle.load(f)
        self.data = save_dict['data']
        self.size = save_dict['size']
        self._traj_counter = save_dict['_traj_counter']
        self._start = save_dict['_start']
        self.traj_bounds = save_dict['traj_bounds']
