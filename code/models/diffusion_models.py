import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ActMLPDiffusion(nn.Module):

    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device='cuda',
                 hidden_dim=256):

        super(ActMLPDiffusion, self).__init__()
        self.device = device
        self.state_dim = state_dim
        t_dim = 64
        self.t_dim = t_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim * 11 + action_dim + t_dim + 11
        
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish())

        self.final_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, state_dim * 11 + action_dim + 11),)
                                     

    def forward(self, x, cond, time):
        # input shape: 
        # x: [B*seq_len, state_dim * 2 + action_dim]
        # time: [B*seq_len, 1]
        bs = cond.shape[0]
        x = x.reshape(bs, -1, 11 * (self.state_dim + 1) + self.action_dim)
       
        #print(x.shape, cond.shape,time.shape)
        x[:,:, :cond.shape[-1]] == cond
        
        t = self.time_mlp(time).reshape(bs, -1, self.t_dim)
        #print(t.shape, x.shape)
        x = torch.cat([x,t], dim=-1)
        x = self.mid_layer(x)
        
        return self.final_layer(x)
    


class RewardMLPDiffusion(nn.Module):
    """
    MLP Model for reward
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device='cuda',
                 hidden_dim=256,
                 t_dim=64):

        super(RewardMLPDiffusion, self).__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, self.t_dim),
        )

        input_dim = state_dim * 11 + 1 + t_dim
        self.state_dim = state_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish())

        self.final_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.Mish(),
                                       nn.Linear(hidden_dim, state_dim * 11 + 1),)
                                     

    def forward(self, x, cond, time):
        # input shape: 
        # x: [B*seq_len, state_dim + 1]
        # time: [B*seq_len, 1]
        # apply conditioning:
        print(x.shape, cond.shape, time.shape)
        shapes = x.shape
        bs = cond.shape[0]
        x = x.reshape(bs, -1, self.state_dim * 11 + 1)
        cond = cond.reshape(bs, x.shape[1], -1)
        print(cond.shape, x.shape)
        x[:,:, :-1] == cond
        t = self.time_mlp(time).reshape(bs, -1, self.t_dim)
        x = torch.cat([x,t], dim=-1)
        x = self.mid_layer(x)
        x = self.final_layer(x)
        x = x.reshape(shapes[0], shapes[1], -1)
        return x