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
    apply_conditioning_rewdiffusion,
    Losses,
)
from prettytable import PrettyTable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import diffuser.utils as utils


from .dt_models import TrajectoryModel, GPT2Model
from .reward_models import RewardMLP, RewardTransformerEncDec, RewardTransformerEncDec_statesonly
from .action_models import ActionMLP, ActionDT, ActionTransformerEncDec
from .state_models import StateTransformer, StatePredictionRNN 
import transformers


import torch
import torch.nn as nn
import torch.nn.functional as F



class Diffusion_model(nn.Module):
    def __init__(self,  model, horizon, observation_dim, action_dim, n_timesteps=1000,
         clip_denoised=False, predict_epsilon=True, loss_discount=1.0, loss_weights=None, 
        condition_guidance_w=0.1, noise_scale=0.5, diffuse_for='action', device='cuda',
        returns_condition=True):
        
        super().__init__()

        self.device = device
        self.diffuse_for = diffuse_for
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.noise_scale = noise_scale
        self.returns_condition = returns_condition
        self.duffse_for = diffuse_for
        self.transition_dim = observation_dim + action_dim
        
        self.sample_t = 0 # used for diffusion warm starting.
        
        self.model = model
            
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount).to(device)
        
        self.loss_fn = Losses['state_l2'](loss_weights)
        self.loss_fn_act_diffusion = Losses['l2']()

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.diffuse_for == 'state':
            if self.returns_condition:
                # epsilon could be epsilon or x0 itself
                epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
                epsilon_uncond = self.model(x, cond, t, returns,force_dropout=True)
                epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
            else:
                epsilon = self.model(x, cond, t)
        elif self.diffuse_for == 'action':
            epsilon = self.model(x, cond, t)
        elif self.diffuse_for == 'reward':
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x = x.reshape(epsilon.shape[0], epsilon.shape[1], -1)
        
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None,):
        
        b, *_, device = *x.shape, x.device
        
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns,)
        x = x.reshape(b, x.shape[1], -1)
        noise = self.noise_scale * torch.randn_like(x).cuda()
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).cuda()
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop_warmup(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = self.noise_scale * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        if self.sample_t % 100 == 0:
            sample_steps = self.n_timesteps
        else:
            sample_steps = 50
        for i in reversed(range(0, sample_steps)):
            
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if self.sample_t % 100 == 0 and i == 150:
                self.stored_diffusion = x
            if self.sample_t > 0 and i == sample_steps-1:
                #x = torch.cat([x[:, :100-self.sample_t%100,:] , self.stored_diffusion[:, 100-self.sample_t%100:,:]], dim=1)
                x = torch.cat([self.stored_diffusion[:, self.sample_t % 100:, :], x[:, -self.sample_t % 100:, :]], dim=1)

            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, 0)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
        self.sample_t += 1
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None,  verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = self.noise_scale * torch.randn(shape, device=device)
        if self.duffse_for == 'action':
            x = apply_conditioning_actdiffusion(x, cond, self.action_dim)
        elif self.diffuse_for == 'reward':
            x = apply_conditioning_rewdiffusion(x, cond, self.action_dim)
        else:
            x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long).cuda()
            x = self.p_sample(x, cond, timesteps, returns)
            if self.duffse_for == 'action':
                x = apply_conditioning_actdiffusion(x, cond, self.action_dim)
            elif self.diffuse_for == 'reward':
                x = apply_conditioning_rewdiffusion(x, cond, self.action_dim)
            else:
                x = apply_conditioning(x, cond, 0)


            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, warmup=False, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        horizon = horizon or self.horizon
        
        device = self.betas.device
        if self.diffuse_for == 'state':
            batch_size = len(cond[0])
            shape = (batch_size, horizon, self.observation_dim)
        elif self.diffuse_for == 'action':
            batch_size = cond.shape[0]
            shape = (batch_size, 11 * self.observation_dim + 11 + self.action_dim)
        elif self.diffuse_for == 'reward':
            batch_size = cond.shape[0]
            shape = (batch_size, self.observation_dim * 11 + 1)
        
        if not warmup:
            return self.p_sample_loop(shape, cond, returns,  *args, **kwargs)
        else:
            return self.p_sample_loop_warmup(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise):
        sample = (
            extract(self.sqrt_alphas_cumprod.cuda(), t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod.cuda(), t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
    
        if self.diffuse_for == 'action':
            B, seq_len, dim = x_start.shape
            x_start = x_start.reshape(B * seq_len, dim)
            noise = torch.randn_like(x_start).cuda()
            x_noisy = self.q_sample(x_start=x_start, t=t.cuda(), noise=noise)
            x_recon = self.model(x=x_noisy, cond=cond, time=t)
            noise = noise.reshape(B, seq_len, dim)
            x_recon = x_recon.reshape(B, seq_len, dim)
        elif self.diffuse_for == 'reward':
            B, seq_len, dim = x_start.shape
            x_start = x_start.reshape(B * seq_len, dim)
            noise = torch.randn_like(x_start).cuda()
            x_noisy = self.q_sample(x_start=x_start, t=t.cuda(), noise=noise)
            x_recon = self.model(x=x_noisy, cond=cond, time=t)
            noise = noise.reshape(B, seq_len, dim)
            x_recon = x_recon.reshape(B, seq_len, dim)
        elif self.diffuse_for == 'state':
            noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_noisy = apply_conditioning(x_noisy, cond, 0)
            x_recon = self.model(x=x_noisy, cond=cond, time=t, returns=returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            if self.diffuse_for == 'action':
                loss = F.mse_loss(x_recon, noise)
            elif self.diffuse_for == 'reward':
                loss = F.mse_loss(x_recon, noise)
            else:
                loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss

    def loss(self, x_start, cond, t, returns=None):
        
        diffuse_loss = self.p_losses(x_start=x_start, cond=cond, t=t, returns=returns)
        return diffuse_loss

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
