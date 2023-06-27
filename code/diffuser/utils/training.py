import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
import gym
from copy import deepcopy
from torch.utils.data import Subset, DataLoader
import random

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from tqdm import tqdm

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class TrainerThreemodule(object):
    def __init__(
        self,
        decision_plugins,
        dataset,
        renderer=None,
        env_name='hopper',
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        eval_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        train_inv=True,
        train_rew=True,
        train_state=False
    ):
        super().__init__()
        self.model = decision_plugins
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints
        self.prefix = env_name

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel
        self.train_inv = train_inv
        self.train_state = train_state
        self.train_rew = train_rew

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        num_samples = len(self.dataset)
        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)

        val_ratio = 0.05
        num_val_samples = int(val_ratio * num_samples)
        # Shuffle the indices of the dataset
        indices = list(range(num_samples))
        random.shuffle(indices)
        # Split the indices into training and validation indices
        train_indices = indices[:-num_val_samples]
        val_indices = indices[-num_val_samples:]
        

        # Create Subset objects for the training and validation sets
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)

        self.train_dataloader = cycle(torch.utils.data.DataLoader(
            train_subset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

        self.eval_dataloader = torch.utils.data.DataLoader(
            val_subset, batch_size=64, num_workers=0, shuffle=False, pin_memory=True
        )

       
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.optimizer = torch.optim.Adam(decision_plugins.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0
        self.best_eval_inv_loss = 99999
        self.best_eval_rew_loss = 99999
        self.best_eval_state_loss = 99999

        self.device = train_device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    def check_previous_model(self):
        checkpoint_dir = os.path.join(self.bucket, self.prefix, 'checkpoint')
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('state_') and 'best' not in f]
            
            
            if checkpoint_files:
                # sort the checkpoint files by step number
                checkpoint_files = sorted(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
                latest_checkpoint_file = checkpoint_files[-1]
                latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
                
                self.load(latest_checkpoint_path)

                start_step = self.step
                print(f'Resuming training from checkpoint {latest_checkpoint_path}, starting from step {start_step}')
            else:
                start_step = 0
        else:
            start_step = 0
        return start_step
    

    def train(self, n_train_steps, writer=None):
        if self.step == 0:
            self.check_previous_model()
        
        for step in tqdm(range(n_train_steps)):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.train_dataloader)
                batch = batch_to_device(batch, device=self.device)
                
                loss, diffuse_loss, inv_loss, rew_loss = self.model.loss(*batch)
                
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                
                if self.train_inv:
                    self.save_inv()
                if self.train_rew:
                    self.save_rew()
                if self.train_state:
                    self.save_state()
                    
            if self.step % 5000 == 0:
                eval_losses = []
                eval_state_losses = []
                eval_inv_losses = []
                eval_reward_losses = []

                with torch.no_grad():
                    for batch in self.eval_dataloader:
                    
                        batch = batch_to_device(batch, device=self.device)
                        
                        eval_loss, eval_state_loss, eval_inv_loss, eval_rew_loss = self.model.loss(*batch)
                        eval_losses.append(eval_state_loss + eval_inv_loss + eval_rew_loss)
                        eval_state_losses.append(eval_state_loss)
                        eval_inv_losses.append(eval_inv_loss)
                        eval_reward_losses.append(eval_rew_loss)
                writer.add_scalar('eval/eval_total_loss', torch.tensor(eval_losses).mean() * 100, self.step)
                writer.add_scalar('eval/eval_inv_loss', torch.tensor(eval_inv_losses).mean() * 100, self.step)
                writer.add_scalar('eval/eval_state_loss', torch.tensor(eval_state_losses).mean() * 100, self.step)
                writer.add_scalar('eval/eval_reward_loss', torch.tensor(eval_reward_losses).mean() * 100, self.step)
                
                if self.train_rew:
                    if torch.tensor(eval_reward_losses).mean() < self.best_eval_rew_loss:
                        self.save_rew('best')
                        self.best_eval_rew_loss = torch.tensor(eval_reward_losses).mean()
                        print('Best eval reward loss:',torch.tensor(eval_reward_losses).mean() * 100,self.step)
                if self.train_inv:
                    if torch.tensor(eval_inv_losses).mean() < self.best_eval_inv_loss:
                        self.save_inv('best')
                        self.best_eval_inv_loss = torch.tensor(eval_inv_losses).mean()
                        print('Best eval action loss:',torch.tensor(eval_inv_losses).mean() * 100,self.step)
                if self.train_state:
                    if torch.tensor(eval_state_losses).mean() < self.best_eval_state_loss:
                        self.save_state('best')
                        self.best_eval_state_loss = torch.tensor(eval_state_losses).mean()
                        print('Best eval state loss:',torch.tensor(eval_state_losses).mean() * 100,self.step)


            if self.step % self.log_freq == 0:
            
                writer.add_scalar('train/loss', loss, self.step)
                writer.add_scalar('train/diffuse_loss', diffuse_loss, self.step)
                writer.add_scalar('train/inv_loss', inv_loss, self.step)
                writer.add_scalar('train/reward_loss', rew_loss, self.step)

            self.step += 1


    def save_state(self, best=None):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_model.state_dict(),
            'ema': self.ema_model.state_model.state_dict()
        }

        savepath = os.path.join(self.bucket, self.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        if best == 'best':
            savepath = os.path.join(savepath, 'state_state_best.pt')
        else:
            savepath = os.path.join(savepath, f'state_state_{self.step}.pt')
        
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')

    def save_inv(self, best=None):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        
        data = {
            'step': self.step,
            'model': self.model.action_model.state_dict(),
            'ema': self.ema_model.action_model.state_dict()
        }
        
        savepath = os.path.join(self.bucket, self.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        if best == 'best':
            savepath = os.path.join(savepath, 'state_inv_best.pt')
        else:
            savepath = os.path.join(savepath, f'state_inv_{self.step}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved action model model to {savepath}')
    
    def save_rew(self, best=None):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.reward_model.state_dict(),
            'ema': self.ema_model.reward_model.state_dict()
        }
        savepath = os.path.join(self.bucket, self.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)

        if best == 'best':
            savepath = os.path.join(savepath, 'state_rew_best.pt')
        else:
            savepath = os.path.join(savepath, f'state_rew_{self.step}.pt')
                
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved reward model to {savepath}')


    def save(self):
        '''
            final save
        '''
        if self.train_state:
            self.save_state()
        if self.train_inv:
            self.save_inv()
        if self.train_rew:
            self.save_rew()

    def load(self, path=None):
        '''
            loads model and ema from disk
        '''
        loadpath = path
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        if self.train_state:
            self.model.state_model.load_state_dict(data['model'])
            self.ema_model.state_model.load_state_dict(data['ema'])
            print('loaded state model from', loadpath)
        if self.train_rew:
            self.model.reward_model.load_state_dict(data['model'])
            self.ema_model.reward_model.load_state_dict(data['ema'])
            print('loaded reward model from', loadpath)
        if self.train_inv:
            self.model.action_model.load_state_dict(data['model'])
            self.ema_model.action_model.load_state_dict(data['ema'])
            print('loaded action model from', loadpath)

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join('images', f'sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, 1), self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

    def inv_render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, 1), self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

