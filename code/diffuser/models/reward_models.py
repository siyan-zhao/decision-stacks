import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Add the positional encodings to the input embeddings
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class RewardTransformerEncDec(nn.Module):
    def __init__(self, observe_dim, reward_dim, hidden_size=128, num_layers=3, num_heads=1, dropout=0.1, max_ep_len=1500):
        super().__init__()
        self.observation_dim = observe_dim
        # Embed observation and rewards
        self.embed_state = torch.nn.Linear(observe_dim, hidden_size)
        self.embed_rewards = torch.nn.Linear(1, hidden_size)
        self.embed_tgt = torch.nn.Linear(reward_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        # Encoder
        self.encoder_embedding = nn.Linear(2 *  hidden_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.transformer = nn.Transformer(d_model = hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          dim_feedforward=128)
        
        self.output_layer = nn.Linear(hidden_size, reward_dim)
        

    def forward(self, states, rtgs, rewards, timesteps=None):
        batch_size, seq_len, _  = states.shape
        _, tgt_len, _ = rewards.shape
        states_time_embeddings = self.embed_timestep(timesteps.to(dtype=torch.long)).reshape(batch_size, seq_len, -1)
        states_output = self.embed_state(states) + states_time_embeddings

        rtg_output = self.embed_rewards(rtgs)
        encoder_input = torch.cat([states_output, rtg_output.unsqueeze(-2).expand(-1, states.shape[-2], -1)], -1)
        encoder_output = self.encoder_embedding(encoder_input).permute(1, 0, 2)
        encoder_output = encoder_output * math.sqrt(encoder_output.size(-1))

        tgt_time_embeddings = self.embed_timestep(timesteps[:, :tgt_len].to(dtype=torch.long)).reshape(batch_size, tgt_len, -1) # with start tokens
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).cuda()
        
        dec_outputs = self.embed_tgt(rewards) + tgt_time_embeddings
        dec_outputs = dec_outputs.permute(1, 0, 2)
        dec_outputs = self.transformer(src=encoder_output, tgt=dec_outputs, src_mask=None,
                                       tgt_mask = tgt_mask, memory_mask=None, src_key_padding_mask=None,
                                        tgt_key_padding_mask=None, memory_key_padding_mask=None )
        output = self.output_layer(dec_outputs.transpose(0,1 ))
        return output
    
    def generate(self, states, tgt_reward, rtg, timestep, max_len=100):
                
        # timesteps [past rewards + cur rew + next rewards]
        # tgt_reward: [past rewards]
        batch_size, seq_len, _  = states.shape
        _, tgt_len, _ = tgt_reward.shape
        # 1. add time embeddings to states and embed states
        states_time_embeddings = self.embed_timestep(timestep.cuda().to(dtype=torch.long)).reshape(batch_size, seq_len, -1)
        states_output = self.embed_state(states) + states_time_embeddings
        rtg_output = self.embed_rewards(rtg)
        encoder_input = torch.cat([states_output, rtg_output.unsqueeze(-2).expand(-1, states.shape[-2], -1)], -1)
        encoder_output = self.encoder_embedding(encoder_input).permute(1, 0, 2)
        encoder_output = encoder_output * math.sqrt(encoder_output.size(-1))

        # Decoder:
        # 1. get time step for decoder.

        tgt_timestep = torch.arange(timestep[0, 0], timestep[0, 0] + tgt_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        # 2. embed tgt and add time embeddings.
        assert not torch.any(torch.isnan(tgt_reward))
        tgt_time_embeddings = self.embed_timestep(tgt_timestep.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1)
        assert not torch.any(torch.isnan(tgt_time_embeddings))
       
        dec_outputs = self.embed_tgt(tgt_reward) + tgt_time_embeddings
        dec_outputs = dec_outputs.permute(1, 0, 2)

        # Generate:
     
        predicted_r = tgt_reward

        for i in range(max_len):
            if i != 0:
                tgt_len += 1
                tgt_reward = torch.cat([tgt_reward, output.reshape(-1, 1,1)], dim=-2)
                tgt_timestep = torch.arange(timestep[0, 0], timestep[0, 0] + tgt_len).unsqueeze(0).repeat(batch_size, 1).cuda()
                tgt_time_embeddings = self.embed_timestep(tgt_timestep.to(dtype=torch.long)).reshape(batch_size, tgt_len, -1)
                dec_outputs = self.embed_tgt(tgt_reward) + tgt_time_embeddings
                dec_outputs = dec_outputs.permute(1, 0, 2)
            #tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).cuda()
            this_dec_outputs = self.transformer(src=encoder_output, tgt=dec_outputs, src_mask=None,
                                        tgt_mask = None, memory_mask=None, src_key_padding_mask=None,
                                            tgt_key_padding_mask=None, memory_key_padding_mask=None ) # tgt len, B, hidden
            
            output = self.output_layer(this_dec_outputs.transpose(0, 1 )[:, -1]).reshape(batch_size, 1, 1)
            predicted_r = torch.cat([predicted_r, output], dim=-2)




        return tgt_reward, predicted_r
