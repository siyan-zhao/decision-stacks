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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings in advance
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is assumed to be of shape [batch_size, sequence_length] and contain timestep values
        positional_encodings = self.pe[:, x, :]
        return positional_encodings


class ActionMLP(nn.Module):
    def __init__(self, observation_dim, act_dim, reward_dim=1, goal_dim=1, seq_len=100, hidden_dim=128, dropout_prob=0.1):
        super(ActionMLP, self).__init__()
        self.input_size = observation_dim * seq_len #+ goal_dim + reward_dim * seq_len
        #self.input_size = 2 * observation_dim
       # self.output_dim = act_dim * seq_len
        self.output_dim = seq_len * act_dim
        self.act_dim = act_dim
        self.rew_dim = reward_dim
        self.seq_len = seq_len

        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x, rewards, goal, timestep=None):
        # x: input sequence of observation frames (batch_size, seq_len, observation_dim)
        # goal: goal vector (batch_size, goal_dim)

        batch_size, seq_len, observation_dim = x.size()

        # flatten the input sequence
        x = x.reshape(batch_size, seq_len * observation_dim)
        #rewards = rewards.reshape(batch_size, seq_len * self.rew_dim)

        #assert seq_len == self.seq_len

        #obs_reward_goal = torch.cat([x, rewards, goal], dim=-1)
        
        #actions = self.fc_layers(obs_reward_goal)
        actions = self.fc_layers(x)
        actions = actions.reshape(batch_size, seq_len, self.act_dim)

        return actions


class ActionTransformerEncDec(nn.Module):
    def __init__(self, observe_dim, action_dim, hidden_size=128, num_layers=3, num_heads=4, dropout=0.1, max_ep_len=1000):
        super().__init__()
        self.observation_dim = observe_dim
        self.action_dim = action_dim
        # Embed observation and rewards
        self.embed_state = torch.nn.Linear(observe_dim, hidden_size)
        self.embed_rewards = torch.nn.Linear(1+1, hidden_size)
        self.embed_tgt = torch.nn.Linear(action_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # Encoder
        self.encoder_embedding = nn.Linear(2 * hidden_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.transformer = nn.Transformer(d_model = hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dim_feedforward=1024)
        
        self.output_layer = nn.Linear(hidden_size, action_dim)
        

    def forward(self, states, rewards, rtgs, actions, timesteps=None):
        # P(a|o, r, T)

        batch_size, seq_len, _  = states.shape
        _, tgt_len, _ = actions.shape
        states_time_embeddings = self.embed_timestep(timesteps.to(dtype=torch.long)).reshape(batch_size, seq_len, -1)

        state_embed = self.embed_state(states) + states_time_embeddings
        reward_embed = self.embed_rewards(torch.cat((rewards, rtgs.unsqueeze(-1).expand(-1, rewards.shape[-2], -1)), dim=-1)) + states_time_embeddings# concatenate rtgs and rewards:
        encoder_input = torch.cat([state_embed, reward_embed], dim=-1)
        encoder_output = self.encoder_embedding(encoder_input)
        encoder_output = encoder_output.permute(1, 0, 2)

        encoder_output = encoder_output * math.sqrt(encoder_output.size(-1))
        tgt_time_embeddings = self.embed_timestep(timesteps[:, :tgt_len].to(dtype=torch.long)).reshape(batch_size, tgt_len, -1) # with start tokens
        
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).cuda()
        dec_outputs = self.embed_tgt(actions) + tgt_time_embeddings
        dec_outputs = dec_outputs.permute(1, 0, 2)
        
        dec_outputs = self.transformer(src=encoder_output, tgt=dec_outputs, src_mask=None,
                                       tgt_mask = tgt_mask, memory_mask=None, src_key_padding_mask=None,
                                        tgt_key_padding_mask=None, memory_key_padding_mask=None )
        output = self.output_layer(dec_outputs.transpose(0, 1 ))
        
        return output
    
    def generate(self, states, rewards, rtg, tgt_action, timestep, max_len=100, gt_action=None):
        # rtg: return of the trajectory
        
        batch_size, seq_len, _  = states.shape
        _, tgt_len, _ = tgt_action.shape
        # 1. add time embeddings to states and embed states

        states_time_embeddings = self.embed_timestep(timestep.cuda().to(dtype=torch.long)).reshape(batch_size, seq_len, -1)
        state_embed = self.embed_state(states) + states_time_embeddings
        # 2. add time embeddings to rewards and embed rewards
        reward_embed = self.embed_rewards(torch.cat((rewards, rtg.unsqueeze(-1).expand(-1, rewards.shape[-2], -1)), dim=-1)) + states_time_embeddings# concatenate rtgs and rewards:
        # 3. concat reward and states to get input to encoder
        encoder_input = torch.cat([state_embed, reward_embed], dim=-1)
        encoder_output = self.encoder_embedding(encoder_input).permute(1, 0, 2)
        encoder_output = encoder_output * math.sqrt(encoder_output.size(-1))

        
        # Decoder:
        # 1. get time step for decoder.
        tgt_timestep = torch.arange(timestep[0, 0], timestep[0, 0] + tgt_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        # 2. embed tgt and add time embeddings.
        assert not torch.any(torch.isnan(tgt_action))
        tgt_time_embeddings = self.embed_timestep(tgt_timestep.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1)
        assert not torch.any(torch.isnan(tgt_time_embeddings))
        
        dec_outputs = self.embed_tgt(tgt_action) + tgt_time_embeddings
        dec_outputs = dec_outputs.permute(1, 0, 2)

        # Generate:
        predicted_a = tgt_action
        for i in range(max_len-1):
           
            this_dec_outputs = self.transformer(src=encoder_output, tgt=dec_outputs, src_mask=None,
                                        tgt_mask = None, memory_mask=None, src_key_padding_mask=None,
                                            tgt_key_padding_mask=None, memory_key_padding_mask=None ) # tgt len, B, hidden

            output = self.output_layer(this_dec_outputs.transpose(0, 1 )[:, -1]).reshape(batch_size, 1, self.action_dim)
            
            tgt_len += 1
           
            tgt_action = torch.cat([tgt_action, output.reshape(-1, 1, self.action_dim)], dim=-2)
            predicted_a = torch.cat([predicted_a, output], dim=-2)

            tgt_timestep = torch.arange(timestep[0, 0], timestep[0, 0] + tgt_len).unsqueeze(0).repeat(batch_size, 1).cuda()
            
            tgt_time_embeddings = self.embed_timestep(tgt_timestep.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1)
            dec_outputs = self.embed_tgt(tgt_action) + tgt_time_embeddings
            dec_outputs = dec_outputs.permute(1, 0, 2)

        return tgt_action, predicted_a
        
class ActionTransformerEncDec_prev(nn.Module):
    def __init__(self, observe_dim, action_dim, hidden_size=128, num_layers=3, num_heads=4, dropout=0.1, max_ep_len=1000):
        super().__init__()
        self.observation_dim = observe_dim
        self.action_dim = action_dim
        # Embed observation and rewards
        self.embed_state = torch.nn.Linear(observe_dim, hidden_size)
        self.embed_rewards = torch.nn.Linear(1+1, hidden_size)
        self.embed_tgt = torch.nn.Linear(action_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # Encoder
        self.encoder_embedding = nn.Linear(2 * hidden_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.transformer = nn.Transformer(d_model = hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dim_feedforward=1024)
        
        self.output_layer = nn.Linear(hidden_size, action_dim)
        

    def forward(self, states, rewards, rtgs, actions, timesteps=None):

        batch_size, seq_len, _  = states.shape
        _, tgt_len, _ = actions.shape
        states_time_embeddings = self.embed_timestep(timesteps.to(dtype=torch.long)).reshape(batch_size, seq_len, -1)

        state_embed = self.embed_state(states) + states_time_embeddings
        reward_embed = self.embed_rewards(torch.cat((rewards, rtgs.unsqueeze(-1).expand(-1, rewards.shape[-2], -1)), dim=-1)) + states_time_embeddings# concatenate rtgs and rewards:
        encoder_input = torch.cat([state_embed, reward_embed], dim=-1)
        encoder_output = self.encoder_embedding(encoder_input)
        encoder_output = encoder_output.permute(1, 0, 2)

        encoder_output = encoder_output * math.sqrt(encoder_output.size(-1))
        tgt_timesteps = timesteps - 1
        
        tgt_time_embeddings = self.embed_timestep(tgt_timesteps.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1) # with start tokens
        
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).cuda()

        dec_outputs = self.embed_tgt(actions) + tgt_time_embeddings
        dec_outputs = dec_outputs.permute(1, 0, 2)
        
        dec_outputs = self.transformer(src=encoder_output, tgt=dec_outputs, src_mask=None,
                                       tgt_mask = tgt_mask, memory_mask=None, src_key_padding_mask=None,
                                        tgt_key_padding_mask=None, memory_key_padding_mask=None )
        output = self.output_layer(dec_outputs.transpose(0, 1 ))
        
        return output
    
    def generate(self, states, rewards, rtg, tgt_action, timestep, max_len=100, gt_action=None):
        # rtg: return of the trajectory

        batch_size, seq_len, _  = states.shape
        _, tgt_len, _ = tgt_action.shape
        # 1. add time embeddings to states and embed states
        inputtimestep = timestep + 1

        states_time_embeddings = self.embed_timestep(inputtimestep.cuda().to(dtype=torch.long)).reshape(batch_size, seq_len, -1)
        state_embed = self.embed_state(states) + states_time_embeddings
        # 2. add time embeddings to rewards and embed rewards
        reward_embed = self.embed_rewards(torch.cat((rewards, rtg.unsqueeze(-1).expand(-1, rewards.shape[-2], -1)), dim=-1)) + states_time_embeddings# concatenate rtgs and rewards:
        # 3. concat reward and states to get input to encoder
        encoder_input = torch.cat([state_embed, reward_embed], dim=-1)
        encoder_output = self.encoder_embedding(encoder_input).permute(1, 0, 2)
        encoder_output = encoder_output * math.sqrt(encoder_output.size(-1))

        # Decoder:
        # 1. get time step for decoder.
        tgt_timestep = torch.arange(timestep[0, 0], timestep[0, 0] + tgt_len).unsqueeze(0).repeat(batch_size, 1).cuda()


        # 2. embed tgt and add time embeddings.
        assert not torch.any(torch.isnan(tgt_action))
        tgt_time_embeddings = self.embed_timestep(tgt_timestep.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1)
        assert not torch.any(torch.isnan(tgt_time_embeddings))

        dec_outputs = self.embed_tgt(tgt_action) + tgt_time_embeddings
        dec_outputs = dec_outputs.permute(1, 0, 2)

        # Generate:
        predicted_a = tgt_action
        for i in range(max_len-1):
            
            this_dec_outputs = self.transformer(src=encoder_output, tgt=dec_outputs, src_mask=None,
                                        tgt_mask = None, memory_mask=None, src_key_padding_mask=None,
                                            tgt_key_padding_mask=None, memory_key_padding_mask=None ) # tgt len, B, hidden

            output = self.output_layer(this_dec_outputs.transpose(0, 1 )[:, -1]).reshape(batch_size, 1, self.action_dim)
            
            tgt_len += 1
            tgt_action = torch.cat([tgt_action, output.reshape(-1, 1, self.action_dim)], dim=-2)
            predicted_a = torch.cat([predicted_a, output], dim=-2)
            tgt_timestep = torch.arange(timestep[0, 0], timestep[0, 0] + tgt_len).unsqueeze(0).repeat(batch_size, 1).cuda()
            tgt_time_embeddings = self.embed_timestep(tgt_timestep.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1)
            dec_outputs = self.embed_tgt(tgt_action) + tgt_time_embeddings
            dec_outputs = dec_outputs.permute(1, 0, 2)

        return tgt_action, predicted_a


class ActionTransformerEncDec_woreward(nn.Module):
    def __init__(self, observe_dim, action_dim, hidden_size=128, num_layers=3, num_heads=4, dropout=0.1, max_ep_len=1000):
        super().__init__()
        self.observation_dim = observe_dim
        self.action_dim = action_dim
        # Embed observation and rewards
        self.embed_state = torch.nn.Linear(observe_dim, hidden_size)
        self.embed_rewards = torch.nn.Linear(1, hidden_size)
        self.embed_tgt = torch.nn.Linear(action_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # Encoder
        self.encoder_embedding = nn.Linear(2 * hidden_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.transformer = nn.Transformer(d_model = hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dim_feedforward=1024)
        
        self.output_layer = nn.Linear(hidden_size, action_dim)
        

    def forward(self, states, rtgs, actions, timesteps=None):
        # P(a|o, T)
        batch_size, seq_len, _  = states.shape
        _, tgt_len, _ = actions.shape
        timesteps = timesteps.unsqueeze(1)

        states_time_embeddings = self.embed_timestep(timesteps.to(dtype=torch.long)).reshape(batch_size, seq_len, -1)
        state_embed = self.embed_state(states) + states_time_embeddings
        reward_embed = self.embed_rewards(rtgs.unsqueeze(-1).repeat(1, seq_len, 1)) + states_time_embeddings# concatenate rtgs and rewards:
        encoder_input = torch.cat([state_embed, reward_embed], dim=-1)
        encoder_output = self.encoder_embedding(encoder_input)
        encoder_output = encoder_output.permute(1, 0, 2)
        encoder_output = encoder_output * math.sqrt(encoder_output.size(-1))
        
        tgt_time_embeddings = self.embed_timestep(timesteps[:, :, :tgt_len].to(dtype=torch.long)).reshape(batch_size, tgt_len, -1) # with start tokens
        
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).cuda()
        dec_outputs = self.embed_tgt(actions) + tgt_time_embeddings
        dec_outputs = dec_outputs.permute(1, 0, 2)
        
        dec_outputs = self.transformer(src=encoder_output, tgt=dec_outputs, src_mask=None,
                                       tgt_mask = tgt_mask, memory_mask=None, src_key_padding_mask=None,
                                        tgt_key_padding_mask=None, memory_key_padding_mask=None )
        output = self.output_layer(dec_outputs.transpose(0, 1))
        
        return output
    
    def generate(self, states, rewards, rtg, tgt_action, timestep, max_len=100, gt_action=None):
        # rtg: return of the trajectory
        
        batch_size, seq_len, _  = states.shape
        _, tgt_len, _ = tgt_action.shape
        # 1. add time embeddings to states and embed states
        states_time_embeddings = self.embed_timestep(timestep.cuda().to(dtype=torch.long)).reshape(batch_size, seq_len, -1)
        state_embed = self.embed_state(states) + states_time_embeddings
        # 2. add time embeddings to rewards and embed rewards
        reward_embed = self.embed_rewards(rtg.unsqueeze(-1).repeat(1, seq_len, 1)) + states_time_embeddings# concatenate rtgs and rewards:
        # 3. concat reward and states to get input to encoder
        encoder_input = torch.cat([state_embed, reward_embed], dim=-1)
        encoder_output = self.encoder_embedding(encoder_input).permute(1, 0, 2)
        encoder_output = encoder_output * math.sqrt(encoder_output.size(-1))
        
        # Decoder:
        # 1. get time step for decoder.
        timestep += 0
        tgt_timestep = torch.arange(timestep[0, 0], timestep[0, 0] + tgt_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        # 2. embed tgt and add time embeddings.
        assert not torch.any(torch.isnan(tgt_action))
        
        tgt_time_embeddings = self.embed_timestep(tgt_timestep.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1)
        assert not torch.any(torch.isnan(tgt_time_embeddings))
        dec_outputs = self.embed_tgt(tgt_action) + tgt_time_embeddings
        dec_outputs = dec_outputs.permute(1, 0, 2)

        # Generate:
        predicted_a = tgt_action
        for i in range(max_len-1):
            this_dec_outputs = self.transformer(src=encoder_output, tgt=dec_outputs, src_mask=None,
                                        tgt_mask = None, memory_mask=None, src_key_padding_mask=None,
                                            tgt_key_padding_mask=None, memory_key_padding_mask=None ) # tgt len, B, hidden
            output = self.output_layer(this_dec_outputs.transpose(0, 1 )[:, -1]).reshape(batch_size, 1, self.action_dim)
            
            tgt_len += 1
            
            tgt_action = torch.cat([tgt_action, output.reshape(-1, 1, self.action_dim)], dim=-2)
            predicted_a = torch.cat([predicted_a, output], dim=-2)     
            tgt_timestep = torch.arange(timestep[0, 0], timestep[0, 0] + tgt_len).unsqueeze(0).repeat(batch_size, 1).cuda()
            tgt_time_embeddings = self.embed_timestep(tgt_timestep.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1)
            dec_outputs = self.embed_tgt(tgt_action) + tgt_time_embeddings
            dec_outputs = dec_outputs.permute(1, 0, 2)
        return tgt_action, predicted_a


