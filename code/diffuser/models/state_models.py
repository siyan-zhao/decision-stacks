import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses)

from .dt_models import TrajectoryModel, GPT2Model
import transformers

import torch
import torch.nn as nn
import math

class StateTransformer(TrajectoryModel):

    """

    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            state_dim,
            hidden_size,
            max_length=None,
            max_ep_len=1000,
            **kwargs
    ):
        super().__init__(state_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.max_length = 1000
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim+1, hidden_size)
        
       
       
        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        

    def forward(self, states, condition_return, timesteps, attention_mask=None):
        '''
        This model is modeling the states trajectory of tau = {R, s1, R, s2, R, s3, ...}, where R is the conditioned total return.
        '''
        
        batch_size, seq_length = states.shape[0], states.shape[1]
       
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to('cuda')
        
        # embed each modality with a different head
        
        states = torch.cat([states, condition_return.reshape(batch_size, 1, 1).repeat(1, seq_length, 1).cuda()], dim=-1)
        state_embeddings = self.embed_state(states)
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = state_embeddings + time_embeddings
        input_embeddings = self.embed_ln(state_embeddings)

        transformer_outputs = self.transformer(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        state_preds = self.predict_state(x)
        return state_preds

    def get_state(self, states, timesteps, condition_return, future_steps):
        B = states.shape[0]
        
        states = states.reshape(B, -1, self.state_dim)
        returns_to_go = condition_return.unsqueeze(-1).reshape(B, 1, 1)
        timesteps = timesteps.reshape(B, -1)
        cur_t = timesteps[0, -1]
        generate_states = []
        for ft in range(future_steps):
            if self.max_length is not None:
                states = states[:,-self.max_length:]
                returns_to_go = returns_to_go[:,-self.max_length:]
                timesteps = timesteps[:,-self.max_length:]

                # pad all tokens to sequence length
                attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])]).unsqueeze(0)
                attention_mask = attention_mask.repeat(B, 1)
                attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(B, -1)
                states = torch.cat(
                    [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                    dim=1).to(dtype=torch.float32)
                timesteps = torch.cat(
                    [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                    dim=1
                ).to(dtype=torch.long)
            else:
                
                attention_mask = None
            
            states_preds = self.forward(
                states, condition_return=returns_to_go, timesteps=timesteps, attention_mask=attention_mask)
            
            generate_states.append(states_preds[:, -1].reshape(B,1,-1))
            states = torch.cat([states, states_preds[:, -1].reshape(B, 1, -1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((B, 1), device='cuda', dtype=torch.long) * (cur_t + ft +1)], dim=1)

        return torch.cat(generate_states, dim=1)
