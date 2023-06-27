from collections import namedtuple
import numpy as np
import torch
import pdb
import random

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset, sequence_dataset_occluded
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
RewardBatch_threemodules = namedtuple('Batch', 'trajectories, conditions, returns, timesteps, this_seq_rewards')
maze_batch = namedtuple('Batch', 'trajectories, conditions, goals, timesteps, this_seq_rewards, total_rewards')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

# obtained from dataset
per_reward_scale = { # & max reward in dataset
            'hopper-medium-replay-v2': 7, # max: 6.385763
            'walker2d-medium-replay-v2': 9, # max: 8.55314
            'halfcheetah-medium-replay-v2':8, # max: 7.61941
            'walker2d-medium-v2': 10, # max: 8.469034
            'halfcheetah-medium-v2': 10, # max: 8.326745
            'hopper-medium-v2': 7, # max: 5.9441
            'hopper-medium-expert-v2': 8, # max: 6.628322
            'walker2d-medium-expert-v2': 10, # max: 8.469034
            'halfcheetah-medium-expert-v2': 15, # max: 13.854624
            'maze2d-umaze-v1':1,
            'maze2d-medium-v1':1,
            'maze2d-large-v1':1
        }

class SequenceDataset_pomdp(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False,
        eval=False, buffer_size=1000, buffer_initialize=True,  dataset='training'):
        
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.dataset = dataset
        self.env_name = env
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset_occluded(env, self.preprocess_fn, env_name=self.env_name)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()


        print(fields)
        plot_rewards = 0
        if plot_rewards:
                import matplotlib.pyplot as plt
                all_rewards = self.fields.rewards
                # Compute the accumulated rewards over time for each trajectory
                accum_rewards = np.cumsum(all_rewards, axis=1)

                # Compute the per-step average reward over all trajectories
                avg_reward = np.mean(all_rewards, axis=0)
                var_reward = np.var(all_rewards, axis=0)
                
                # Compute the average accumulated reward over all trajectories
                avg_accum_reward = np.mean(accum_rewards, axis=0)

                # Plot the per-step average reward
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                ax1.plot(self.env.get_normalized_score(avg_reward))
                ax1.fill_between(range(len(avg_reward)), self.env.get_normalized_score(avg_reward - np.sqrt(var_reward)).ravel(), 
                self.env.get_normalized_score(avg_reward + np.sqrt(var_reward)).ravel(), alpha=0.2)

                ax1.set_title('Per-step Average Reward \n ' + self.env_name)
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Reward')
                var_accum_reward = np.var(accum_rewards, axis=0)
                ax2.plot(self.env.get_normalized_score(avg_accum_reward))
                ax2.fill_between(range(len(avg_accum_reward)), self.env.get_normalized_score(avg_accum_reward - np.sqrt(var_accum_reward)).ravel(), 
                self.env.get_normalized_score(avg_accum_reward + np.sqrt(var_accum_reward)).ravel(), alpha=0.2)

                ax2.set_title('Average Accumulated Reward \n ' + self.env_name)
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Reward')
                
                plt.savefig('reward_graphs' + self.env_name+'.png')
                plt.close()

            

                s = s


        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')


    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):

        indices = self.indices

        return len(indices)

    def __getitem__(self, idx, eps=1e-4):
        
        fields = self.fields
        indices = self.indices

        path_ind, start, end = indices[idx]
        
        observations = fields.normed_observations[path_ind, start:end]
        actions = fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        
        if self.include_returns:
            rewards = fields.rewards[path_ind, start:] 
            this_seq_rewards = fields.rewards[path_ind, start:end] / per_reward_scale[self.env_name]
            
            timesteps = np.arange(start, end).reshape(-1)
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch_threemodules(trajectories, conditions, returns,  timesteps, this_seq_rewards)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False,
        eval=False, buffer_size=1000, buffer_initialize=True,  dataset='training'):
        
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.dataset = dataset
        self.env_name = env
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        print('---------- horizon in dataset:', horizon,'----------')
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        if 'maze' in env.name:
            itr = sequence_dataset_maze2d(env, self.preprocess_fn)
        else:
            itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)


    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):

        indices = self.indices

        return len(indices)

    def __getitem__(self, idx, eps=1e-4):
        
        fields = self.fields
        indices = self.indices
        

        path_ind, start, end = indices[idx]
        
        observations = fields.normed_observations[path_ind, start:end]
        actions = fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        
        if self.include_returns:
            rewards = fields.rewards[path_ind, start:] 
            this_seq_rewards = fields.rewards[path_ind, start:end] / per_reward_scale[self.env_name]
            timesteps = np.arange(start, end).reshape(-1)
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch_threemodules(trajectories, conditions, returns,  timesteps, this_seq_rewards)
        else:
            batch = Batch(trajectories, conditions)

        return batch


class CondSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        t_step = np.random.randint(0, self.horizon)

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        traj_dim = self.action_dim + self.observation_dim

        conditions = np.ones((self.horizon, 2*traj_dim)).astype(np.float32)

        # Set up conditional masking
        conditions[t_step:,:self.action_dim] = 0
        conditions[:,traj_dim:] = 0
        conditions[t_step,traj_dim:traj_dim+self.action_dim] = 1

        if t_step < self.horizon-1:
            observations[t_step+1:] = 0

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }
    def __getitem__(self, idx, eps=1e-4):
        
        fields = self.fields
        indices = self.indices
        

        path_ind, start, end = indices[idx]
        
        observations = fields.normed_observations[path_ind, start:end]
        actions = fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        
        if self.include_returns:
            rewards = fields.rewards[path_ind, start:] 
            this_seq_rewards = fields.rewards[path_ind, start:end] / per_reward_scale[self.env_name]

            timesteps = np.arange(start, end).reshape(-1)
            goals = observations[-1]

            rewards = self.fields['rewards'][path_ind, start:end]
            total_rewards = rewards.sum() / 100
            
            batch = maze_batch(trajectories, conditions, goals, timesteps,  this_seq_rewards, total_rewards)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch