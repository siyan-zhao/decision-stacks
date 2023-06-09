import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    apply_conditioning_actdiffusion,
    Losses,
)
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    #print(table)
   
    return total_params
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import diffuser.utils as utils


from .dt_models import TrajectoryModel, GPT2Model
from .reward_models import *
from .action_models import *
from .state_models import *
from .diffusion_class import Diffusion_model
import transformers


import torch
import torch.nn as nn
import torch.nn.functional as F




class Decision_Stacks(nn.Module):
    def __init__(self,  Config, horizon, observation_dim, action_dim, n_timesteps=1000,
        train_inv=False, train_state=False, train_rew=False, 
        noise_scale=0.5, hidden_dim=256,
        reward_model_name='mlp', state_model_name='diffusion', action_model_name='transformer',
        reward_diffusion_guidance=0.1, action_diffusion_guidance=0.1, state_diffusion_guidance=0.1, wor=False):
        super().__init__()

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.noise_scale = noise_scale
        self.transition_dim = observation_dim + action_dim
        self.n_timesteps = n_timesteps
        self.train_state = train_state
        self.train_inv = train_inv
        self.train_rew = train_rew
        
        self.reward_model_name = reward_model_name
        self.state_model_name = state_model_name
        self.Config = Config
        self.action_model_name = action_model_name
        # reward model choices:
        if self.reward_model_name == 'transformer':
            self.reward_model = RewardTransformerEncDec(observe_dim=observation_dim, reward_dim=1, hidden_size=64).cuda()
        elif self.reward_model_name == 'mlp':
            self.reward_model = nn.Sequential(
                            nn.Linear(self.observation_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1),
                        )
        elif self.reward_model_name == 'diffusion':
            model_config = utils.Config(
                'models.ActionRewardTemporalUnet',
                savepath='model_config.pkl',
                horizon=Config.horizon,
                transition_dim=observation_dim + 1,
                cond_dim=observation_dim + 1,
                dim_mults=Config.dim_mults,
                returns_condition=Config.returns_condition,
                dim=32,
                condition_dropout=Config.condition_dropout,
                calc_energy=Config.calc_energy,
                device=Config.device)
            
            reward_diffusion_model = model_config().cuda()
            self.reward_model = Diffusion_model(reward_diffusion_model, 
                                                horizon=horizon,
                                                observation_dim=observation_dim,
                                                action_dim=action_dim,
                                                n_timesteps=Config.n_diffusion_steps,
                                                clip_denoised=Config.clip_denoised,
                                                predict_epsilon=Config.predict_epsilon,
                                                device=Config.device,
                                                diffuse_for='reward',
                                                condition_guidance_w=reward_diffusion_guidance)
        else:
            raise ValueError("Invalid model choice of reward estimation.")

        # action model choices:
        if self.action_model_name == 'transformer':
            if not wor: 
                self.action_model = ActionTransformerEncDec(observe_dim=observation_dim, action_dim=action_dim).cuda()
            else: # not incorporating rewards
                self.action_model = ActionTransformerEncDec_woreward(observe_dim=observation_dim, action_dim=action_dim).cuda()
        elif self.action_model_name == 'mlp':
            self.action_model = nn.Sequential(
                    nn.Linear(2 * self.observation_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.action_dim),
                ).cuda()
        elif self.action_model_name == 'diffusion':
            model_config = utils.Config(
                'models.ActionRewardTemporalUnet',
                savepath='model_config.pkl',
                horizon=Config.horizon,
                transition_dim=observation_dim + action_dim + 1,
                cond_dim=observation_dim + action_dim + 1,
                dim_mults=Config.dim_mults,
                returns_condition=Config.returns_condition,
                dim=32,
                condition_dropout=Config.condition_dropout,
                calc_energy=Config.calc_energy,
                device=Config.device)
            
            action_diffusion_model = model_config().cuda()
            self.action_model = Diffusion_model(action_diffusion_model, 
                                                horizon=horizon,
                                                observation_dim=observation_dim,
                                                action_dim=action_dim,
                                                n_timesteps=Config.n_diffusion_steps,
                                                clip_denoised=Config.clip_denoised,
                                                predict_epsilon=Config.predict_epsilon,
                                                device=Config.device,
                                                diffuse_for='action',
                                                condition_guidance_w=action_diffusion_guidance)
            
            
        else:
            raise ValueError("Invalid model choice of reward estimation.")
        
        # state model choices:
        if self.state_model_name == 'transformer':
            self.state_model = StateTransformer(state_dim=observation_dim, hidden_size=512, n_layer=4,
                                            n_head=16,
                                            n_inner=4 * 512,
                                            activation_function='relu',
                                            n_positions=1024,
                                            resid_pdrop=0.1,
                                            max_ep_len=1000,
                                            attn_pdrop=0.1).cuda() # the same hyperparameters as Decision Transformer's choice.
        elif self.state_model_name == 'lstm':
            self.state_model = StatePredictionRNN(observation_dim, hidden_dim=512, num_layers=8).to('cuda')
        elif self.state_model_name == 'diffusion':
            model_config = utils.Config(
                'models.StateTemporalUnet',
                savepath='model_config.pkl',
                horizon=Config.horizon,
                transition_dim=observation_dim,
                cond_dim=observation_dim,
                dim_mults=Config.dim_mults,
                dim=Config.dim,
                condition_dropout=Config.condition_dropout,
                calc_energy=Config.calc_energy,
                device=Config.device)
            
            state_model = model_config().cuda()
            self.state_model = Diffusion_model(state_model, 
                                                horizon=horizon,
                                                observation_dim=observation_dim,
                                                action_dim=action_dim,
                                                noise_scale=self.noise_scale,
                                                n_timesteps=Config.n_diffusion_steps,
                                                clip_denoised=Config.clip_denoised,
                                                predict_epsilon=Config.predict_epsilon,
                                                condition_guidance_w=state_diffusion_guidance,
                                                device=Config.device,
                                                diffuse_for='state')
        else:
            raise ValueError("Invalid model choice of reward estimation.")

        if state_model_name != 'diffusion':
            print("Number of parameters in self.state_model:", count_parameters(self.state_model))
        else:
            print("Number of parameters in self.state_model:", count_parameters(self.state_model.model))
       
        if action_model_name != 'diffusion':
            print("Number of parameters in self.action_model:", count_parameters(self.action_model))
        else:
            print("Number of parameters in self.action_model:", count_parameters(self.action_model.model))
        
        if reward_model_name != 'diffusion':
            print("Number of parameters in self.reward_model:", count_parameters(self.reward_model))
        else:
            print("Number of parameters in self.reward_model:", count_parameters(self.reward_model.model))
        print("--------------------------------------------------------")

            
    def loss(self, x, cond, returns=None, timesteps=None, rewards=None, total_reward=None):
        # note that: action and rewards are padded with zeros for transformer input conveniency.
        batch_size = len(x) 
        
        # 1. state model
        inv_loss = float(torch.tensor(0))
        reward_loss = float(torch.tensor(0))
        loss = float(torch.tensor(0))
        state_loss = float(torch.tensor(0))


        observations = x[:, :, self.action_dim:]
        horizon = x.shape[1]
        if self.train_state:
            if self.state_model_name == 'diffusion':

                t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
                state_loss = self.state_model.loss(x_start=x[:, :, self.action_dim:], cond=cond, t=t, returns=returns)

            elif self.state_model_name == 'transformer':
                target_states = torch.clone(observations)
                predicted_states = self.state_model(observations, returns.unsqueeze(1), timesteps.to(torch.long))
                state_loss = F.mse_loss(predicted_states[:,:-1], target_states[:,1:]) # because predicted states are the next states. We supervise on the next states
            elif self.state_model_name == 'lstm':
                target_states = torch.clone(observations)[:,1:,:].cuda()
                predicted_states = self.state_model(observations.cuda(), returns.unsqueeze(1).cuda())[:,:-1,:]
                state_loss = F.mse_loss(predicted_states, target_states)
        # 2. reward model
        if self.train_rew:
            if self.reward_model_name == 'mlp':
                pre_rewards = self.reward_model(observations).cuda()
                reward_loss = F.mse_loss(pre_rewards, rewards.cuda())
            elif self.reward_model_name == 'transformer':
                pre_rewards = self.reward_model(observations[:,:,:].cuda(), returns.cuda(), rewards[:,:-1].cuda(), timesteps.cuda())
                reward_loss = F.mse_loss(pre_rewards, rewards[:,1:])
            elif self.reward_model_name == 'diffusion':
                observations = x[:, :, self.action_dim:]
                state_rew_pairs = torch.cat([observations, rewards], dim=-1)
                t = torch.randint(0, self.n_timesteps, (batch_size,), device='cuda').long()
                cond = {0:state_rew_pairs[:,0]}
                reward_loss = self.reward_model.loss(x_start=state_rew_pairs, cond=cond, t=t, returns=returns)
                
        # 3. action model
        if self.train_inv:
            if self.action_model_name == 'mlp':
                x_t = x[:, :-1, self.action_dim:]
                a_t = x[:, :-1, :self.action_dim]
                x_t_1 = x[:, 1:, self.action_dim:]
                goals = returns[:,:2].reshape(batch_size, 1, -1).repeat(1, x_t.shape[1], 1)
                x_comb_t = torch.cat([x_t, x_t_1, goals], dim=-1)
                x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim + 2)
                pred_a_t = self.action_model(x_comb_t)
                a_t = a_t.reshape(-1, self.action_dim)
                inv_loss = F.mse_loss(pred_a_t, a_t)
            elif self.action_model_name == 'diffusion':
                actions = x[:, :, :self.action_dim] # 1,2,3,4,5,6,...
                observations = x[:, :, self.action_dim:] # 1,2,3,4,5,6,...
                # rewards = 0, 1, 2, 3, 4, 5, 6 ,7
                diffuse_over = torch.cat([observations[:,:], rewards[:,:], actions[:,:]], dim=-1)
                
                cond = {0: diffuse_over[:,0]}
                t = torch.randint(0, self.n_timesteps, (batch_size,), device='cuda').long()
                inv_loss = self.action_model.loss(x_start=diffuse_over, cond=cond, t=t, returns=returns)
            elif self.action_model_name == 'transformer':
                a_t = x[:, :, :self.action_dim].cuda()
                observations = x[:, :, self.action_dim:].cuda()
                a_t = x[:, :, :self.action_dim].cuda()
                start_a_token = torch.zeros((batch_size, 1, self.action_dim)).cuda()
                a_t = torch.cat([start_a_token, a_t], dim=-2)
                pred_a_t = self.action_model(observations.cuda(), rewards.cuda(), returns.cuda(), a_t[:,:-1,].cuda(), timesteps[:,:].cuda())
                pred_a_t = pred_a_t.reshape(-1, self.action_dim)
                a_t_label = a_t[:, 1:]
                
                inv_loss = F.mse_loss(pred_a_t, a_t_label.reshape(-1, self.action_dim))
               
        loss = (1 / 3) * (state_loss + inv_loss + reward_loss)

        return loss, state_loss, inv_loss, reward_loss

