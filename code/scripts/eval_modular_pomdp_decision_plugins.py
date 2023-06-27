import diffuser.utils as utils

import torch
from copy import deepcopy
import numpy as np
import os
import gym
import argparse
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
import time
import torch.nn.functional as F
from config.locomotion_config_decision_plugins import Config

from trajectory.plan import TT_Trajectory_Prediction
import csv
import psutil
def filter_keys(state_dict, prefix='reward_model.'):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = value
    return new_state_dict

per_reward_scale = { # & max reward in dataset
            'hopper-medium-replay-v2': 7, # max: 6.385763
            'walker2d-medium-replay-v2': 9, # max: 8.55314
            'halfcheetah-medium-replay-v2':8, # max: 7.61941
            'walker2d-medium-v2': 10, # max: 8.469034
            'halfcheetah-medium-v2': 10, # max: 8.326745
            'hopper-medium-v2': 7, # max: 5.9441
            'hopper-medium-expert-v2': 8, # max: 6.628322
            'hopper-expert-v2': 8, # max: 6.628322
            'walker2d-medium-expert-v2': 10, # max: 8.469034
            'halfcheetah-medium-expert-v2': 15, # max: 13.854624
        }

def get_timesteps(t, obs_horizon, horizon, num_eval, max_path_length=999):
    if t > obs_horizon:
        timesteps = torch.arange(max(t - obs_horizon, 1), min(t + horizon + 1, max_path_length)).unsqueeze(0).repeat(num_eval, 1)
    else:
        timesteps = torch.arange(0, min(t + horizon + 1, max_path_length)).unsqueeze(0).repeat(num_eval, 1)
    return timesteps

def evaluate(args):
    data_dir = '/home/data/ckpts/siyanz/dd_bayesian/'
    from config.locomotion_config_decision_plugins import Config
    Config.dataset = args.env_name
    if 'hopper' in Config.dataset:
        Config.returns_scale = 400.0
    elif 'halfcheetah' in Config.dataset:
        Config.returns_scale = 1200.0
    elif 'walker' in Config.dataset:
        Config.returns_scale = 550.0 # Determined using rewards from the dataset

    Config.reward_model_name = args.reward_model_name
    Config.action_model_name = args.action_model_name
    Config.state_model_name = args.state_model_name
    
    mlp_action = (args.action_model_name == 'mlp')
   
    env_name = args.env_name
    Config.test_ret = args.testrt
    Config.hidden_dim = 256
    conditional_w = args.conditional_w
    total_done = 0

    """ ------- LOAD MODELS ------- """
    state_folder_name =  args.statefolder
    state_ckpt = args.stateckpt
    path_state = data_dir + state_folder_name + '/' + env_name + '/checkpoint/' + state_ckpt
    #state_folder_name = 'decision_transformer_statemodel'
    #state_ckpt = '10.pt'
    #path_state = '/home/data/ckpts/siyanz/dd_bayesian/' + state_folder_name + '/' + state_ckpt
    #old_state_dict = torch.load(path_state, map_location=Config.device)

    action_folder_name = args.actfolder
    action_ckpt = 'state_inv_' + args.actckpt
    path_action = data_dir +action_folder_name + '/' + env_name + '/checkpoint/' + action_ckpt

    rew_folder_name = args.rewfolder
    rew_ckpt = 'state_rew_' + args.rewckpt
    path_rew = data_dir +rew_folder_name + '/' + env_name + '/checkpoint/' + rew_ckpt
    rewmodel_conditioning = False
    
    Config.hidden_dim =256
    # load dictionaries
    if args.action_model_name == 'mlp':
        act_dict = filter_keys(torch.load(path_action, map_location=Config.device)['model'], prefix='inv_model.')
        ema_act_dict = filter_keys(torch.load(path_action, map_location=Config.device)['ema'], prefix='inv_model.')
    elif args.action_model_name == 'transformer':
        act_dict = torch.load(path_action, map_location=Config.device)['model']
        ema_act_dict = torch.load(path_action, map_location=Config.device)['ema']
    elif args.action_model_name == 'diffusion':
        act_dict = filter_keys(torch.load(path_action, map_location=Config.device)['model'], prefix='model.')
        ema_act_dict = filter_keys(torch.load(path_action, map_location=Config.device)['ema'], prefix='model.')

    if args.state_model_name == 'transformer':
        state_dict = torch.load(path_state, map_location=Config.device)['model']
        ema_state_dict = torch.load(path_state, map_location=Config.device)['ema']
        #state_dict = torch.load(path_state, map_location=Config.device)#['model']
        #ema_state_dict = torch.load(path_state, map_location=Config.device)#['ema']
    elif args.state_model_name == 'diffusion':
        state_dict = filter_keys(torch.load(path_state, map_location=Config.device)['model'], prefix='model.')
        ema_state_dict = filter_keys(torch.load(path_state, map_location=Config.device)['ema'], prefix='model.')
    elif args.state_model_name == 'lstm':
        state_dict = torch.load(path_state, map_location=Config.device)['model']
        ema_state_dict = torch.load(path_state, map_location=Config.device)['ema']

    if args.reward_model_name == 'mlp':
        rew_dict = torch.load(path_rew, map_location=Config.device)['model']
        ema_rew_dict = torch.load(path_rew, map_location=Config.device)['ema']
    elif args.reward_model_name == 'transformer':
        rew_dict = torch.load(path_rew, map_location=Config.device)['model']
        ema_rew_dict = torch.load(path_rew, map_location=Config.device)['ema']
    elif args.reward_model_name == 'diffusion':
        rew_dict = filter_keys(torch.load(path_rew, map_location=Config.device)['model'], prefix='model.')
        ema_rew_dict = filter_keys(torch.load(path_rew, map_location=Config.device)['ema'], prefix='model.')

    
    """ ------- LOAD MODELS ENDS ------- """
   
    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    horizon = Config.horizon
    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )

    
    torch.set_num_threads(1)
    dataset = dataset_config()
   

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    transition_dim = observation_dim

    model_config = utils.Config(
        'models.StateTemporalUnet',
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=observation_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        returns_condition=Config.returns_condition,
        dim=Config.dim,
        condition_dropout=Config.condition_dropout,
        calc_energy=Config.calc_energy,
        device=Config.device,
    )

    decision_plugins_config = utils.Config(
            'models.Decision_Plugins',
            savepath='diffusion_config.pkl',
            Config=Config,
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            hidden_dim=Config.hidden_dim,
            train_inv=Config.train_inv,
            train_state = Config.train_state,
            train_rew = Config.train_rew,
            noise_scale=args.noise_scale,
            reward_model_name=Config.reward_model_name,
            state_model_name=Config.state_model_name,
            action_model_name=Config.action_model_name,
            reward_diffusion_guidance=args.conditional_w,
            action_diffusion_guidance=args.conditional_w, 
            state_diffusion_guidance=args.conditional_w,
        )

    trainer_config = utils.Config(
        utils.TrainerThreemodule,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        env_name = Config.dataset,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
        train_inv=Config.train_inv,
        train_state = Config.train_state,
        train_rew = Config.train_rew,

    )

    decision_plugins = decision_plugins_config()

    trainer = trainer_config(decision_plugins, dataset)
    
    if args.state_model_name == 'diffusion':
        trainer.model.state_model.model.load_state_dict(state_dict)
        trainer.ema_model.state_model.model.load_state_dict(ema_state_dict)
        trainer.ema_model.state_model.model.eval()
    else:
        trainer.model.state_model.load_state_dict(state_dict)
        trainer.ema_model.state_model.load_state_dict(ema_state_dict)
        trainer.ema_model.state_model.eval()
    if args.action_model_name == 'diffusion':
        trainer.model.action_model.model.load_state_dict(act_dict)
        trainer.ema_model.action_model.model.load_state_dict(ema_act_dict)
        trainer.ema_model.action_model.model.eval()
        
    else:
        trainer.model.action_model.load_state_dict(act_dict)
        trainer.ema_model.action_model.load_state_dict(ema_act_dict)
        trainer.ema_model.action_model.eval()

    if args.reward_model_name == 'diffusion':
        trainer.model.reward_model.model.load_state_dict(rew_dict)
        trainer.ema_model.reward_model.model.load_state_dict(ema_rew_dict)
        trainer.ema_model.reward_model.model.eval()
    else:
        trainer.model.reward_model.load_state_dict(rew_dict)
        trainer.ema_model.reward_model.load_state_dict(ema_rew_dict)
        trainer.ema_model.reward_model.eval()

    trainer.ema_model = trainer.ema_model.cuda()

    num_eval = 15
    device = Config.device

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)
    

    if 0:
        if 'halfcheetah' in env_name:
            occlude_start_idx = 8
        if 'hopper' in env_name:
            occlude_start_idx = 5
        if 'walker' in env_name:
            occlude_start_idx = 8
    else:
        # partial eliminate last two velocity dimension
        if 'halfcheetah' in env_name:
            occlude_start_idx = -2
        if 'hopper' in env_name:
            occlude_start_idx = -2
        if 'walker' in env_name:
            occlude_start_idx = -2    
        
    t = 0
   

    obs_list = [env.reset()[None][:, :occlude_start_idx] for env in env_list]
    #obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    avg_scores = []
    std_errs = []
    times = []
    all_rew_mse = []
    horizon = args.future_horizon
    total_done = 0
    obs_horizon = args.obs_horizon
    orig_obs = obs
    while sum(dones) <  num_eval:

        obs = dataset.normalizer.normalize(orig_obs, 'observations')
        
        if t == 0:
            if args.state_model_name == 'diffusion':
                conditions = {0: to_torch(obs, device=device)}
                samples = trainer.ema_model.state_model.conditional_sample(conditions, returns=returns, warmup=args.warmup)[:,:horizon+1]
                all_obs = samples
                samples = samples[:,1:]
            elif args.state_model_name == 'transformer':
                with torch.no_grad():
                    state_transformer_timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1).repeat(num_eval, 1)
                    samples = trainer.ema_model.state_model.get_state(
                        states=torch.from_numpy(obs).float().cuda().reshape(num_eval, 1, -1), 
                        timesteps=state_transformer_timesteps, 
                        condition_return=returns,
                        future_steps=horizon)
                    all_obs = torch.cat([torch.FloatTensor(obs).reshape(num_eval, 1, -1).cuda(), samples], dim=1)

            real_rewards = torch.zeros((num_eval, 0, 1))
            true_obs = torch.zeros((num_eval, 0, observation_dim))
            true_obs = torch.cat([true_obs, torch.from_numpy(obs).reshape(num_eval, 1, -1).to(torch.float32)], dim=1)
        else:
            if args.state_model_name == 'diffusion':
                # apply past observations:
                conditions = {0: to_torch(obs, device=device)}
                #start = max(0, t - obs_horizon)
                #conditions = {this_i: to_torch(true_obs[:, start + this_i]) for this_i in range(min(obs_horizon, t+1))}
                samples = trainer.ema_model.state_model.conditional_sample(conditions, returns=returns, warmup=args.warmup)[:,1:horizon+1]
              
                #samples = samples[:, min(obs_horizon, t) + 1 : horizon + min(obs_horizon, t) + 1]
                
            elif args.state_model_name == 'transformer':
                with torch.no_grad():
                    state_transformer_timesteps = torch.cat(
                        [state_transformer_timesteps[:,-obs_horizon:],
                        torch.ones((num_eval, 1), device=device, dtype=torch.long) * (t)], dim=1)
                    
                    samples = trainer.ema_model.state_model.get_state(
                        states=true_obs[:, -obs_horizon-1:].cuda().reshape(num_eval, 1, -1), 
                        timesteps=state_transformer_timesteps, 
                        condition_return=returns,
                        future_steps=np.minimum(horizon, Config.max_path_length - t-1))
            all_obs = torch.cat([true_obs[:, -obs_horizon-1:].cuda(), samples[:, :np.minimum(horizon, Config.max_path_length - t-2)]], dim=-2)
            
        assert all_obs[0, np.minimum(t, obs_horizon), 0] == true_obs[0, -1, 0]
        act_not_normed_reward_list = ['halfcheetah-medium-v2','walker2d-medium-v2','walker2d-medium-replay-v2']
                
        
        with torch.no_grad():
            # generate time steps:
            # generate time steps:
            if t > obs_horizon:
                timesteps = torch.arange(t - obs_horizon, np.minimum(t + horizon+1, Config.max_path_length-1)).unsqueeze(0).repeat(num_eval, 1)
            else:
                timesteps = torch.arange(0, t + horizon+1).unsqueeze(0).repeat(num_eval, 1)
            
            # for transformers
            if t == 0: # input dummy reward zero at first step
                decoder_actions = torch.zeros((num_eval, 1, action_dim)).cuda()
                decoder_rewards = torch.zeros((num_eval, 1, 1)).cuda()
            else:
            
                if decoder_actions.shape[1] > obs_horizon:
                    decoder_actions = torch.cat([decoder_actions[:, 1:, ], this_t_action.reshape(num_eval, 1, action_dim).cuda()], dim=1).cuda()
                    decoder_rewards = torch.cat([decoder_rewards[:, 1:, ], this_t_reward.reshape(num_eval, 1, 1).cuda()], dim=1).cuda()
                else:
                    decoder_actions = torch.cat([decoder_actions, this_t_action.reshape(num_eval, 1, action_dim).cuda()], dim=1).cuda()
                    decoder_rewards = torch.cat([decoder_rewards, this_t_reward.reshape(num_eval, 1, 1).cuda()], dim=1).cuda()
            if rewmodel_conditioning:
                rew_input = torch.cat([all_obs, returns.reshape(num_eval, 1, 1).repeat(1, all_obs.shape[1], 1)], dim=-1)
            else:
                rew_input = all_obs
            
            if args.reward_model_name == 'mlp':
                syn_rewards = trainer.ema_model.reward_model(rew_input)
            elif args.reward_model_name == 'diffusion':
                cond = {'cond_obs': rew_input, 
                        'cond_rew':real_rewards[:,-obs_horizon:]}
                print(real_rewards.shape, rew_input.shape, real_rewards[:, -obs_horizon:].shape)
                syn_rewards = trainer.ema_model.reward_model.conditional_sample(cond=cond, returns=returns)
                syn_rewards = syn_rewards[:,:rew_input.shape[1], -1].reshape(num_eval, rew_input.shape[1], 1)
                syn_rewards[:,:real_rewards[:, -obs_horizon:].shape[1]] = real_rewards[:, -obs_horizon:]
            elif args.reward_model_name == 'transformer':
                _, syn_rewards = trainer.ema_model.reward_model.generate(states=all_obs, tgt_reward=decoder_rewards, timestep=timesteps, rtg=returns.reshape(num_eval, 1), max_len=horizon)

            if args.action_model_name == 'diffusion':
                # input should be past 5 obs/rewards, cur obs/reward, future 5 obs/rewards

                cond = {'cond_obs': rew_input, 
                        'cond_rew': syn_rewards,
                        'cond_act': decoder_actions[:,1:]}

                pred_actions = trainer.ema_model.action_model.conditional_sample(cond=cond, returns=returns, horizon=Config.horizon)
                this_t_action = pred_actions[:, rew_input.shape[1]-horizon-1, -action_dim:]

                #assert rew_input.shape[1]-horizon-1 == np.minimum(t, obs_horizon)
            elif args.action_model_name == 'transformer':
                
                _, pred_actions = trainer.ema_model.action_model.generate(states=all_obs, rtg=returns.reshape(num_eval, 1), tgt_action=decoder_actions, rewards=syn_rewards, timestep=timesteps, max_len=2)
            
                this_t_action = pred_actions[:, -1]
            elif args.action_model_name == 'mlp':
                # input is [x_t, x_t_next, reward_t, reward_next] 
                obs_comb = torch.cat([true_obs[:,-1].cuda(), samples[:,0], syn_rewards[:,np.minimum(t, obs_horizon)], syn_rewards[:,np.minimum(t, obs_horizon)+1]], dim=-1)

                obs_comb = obs_comb.reshape(-1, 2*observation_dim+2)
                action = trainer.ema_model.action_model(obs_comb)
                this_t_action = action
            action = to_np(this_t_action)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')

        if t == 0:
            normed_observations = samples[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')

        obs_list = []
        this_t_reward = torch.zeros((num_eval, 1, 1))
        this_t_obs = torch.zeros((num_eval, 1, observation_dim))

        for i in range(num_eval):

            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])

            obs_list.append(this_obs[None][:, :occlude_start_idx])
            
            normed_this_obs = dataset.normalizer.normalize(this_obs[:occlude_start_idx],'observations')

            #normed_this_obs = dataset.normalizer.normalize(this_obs,'observations')
            this_t_obs[i] = torch.from_numpy(normed_this_obs)
            if Config.dataset in act_not_normed_reward_list:
                this_t_reward[i] = this_reward
                
            else:
                
                this_t_reward[i] = this_reward  / per_reward_scale[env_name]
                
            #print(this_reward)
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    total_done += 1
                    episode_rewards[i] += this_reward
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += this_reward
                    
        orig_obs = np.concatenate(obs_list, axis=0)

        
        true_obs = torch.cat([true_obs, this_t_obs], dim=-2)
        real_rewards = torch.cat([real_rewards, this_t_reward], dim=1)
        t += 1
        path = '/home/siyanz/decision_plugins/code/scripts/eval_data/eval_pomdp/'+  str(Config.dataset)[:-3]+ '/'  + 'FDP2' + '_w_' + \
                            str(args.conditional_w)  + '_trt'+str(Config.test_ret)  \
                        + 'NS' + str(args.noise_scale) + 'act:'+ action_folder_name + \
                            action_ckpt[4:] + 'st:'+ state_folder_name + state_ckpt[4:] + \
                            'rew:' + rew_folder_name + rew_ckpt[4:] + \
                            'obs_' + str(obs_horizon) + \
                            'future_' + str(horizon) + '.csv'
            
        if t % 2 == 0 or t >= Config.max_path_length-2:
            
            episode_rewards = np.array(episode_rewards)
            normalized_scores = [env_list[i].get_normalized_score(s) for s in episode_rewards]
            avg_score_this = np.mean(normalized_scores)
            
            if 0: # early stop for faster evaluation: not using this if you are doing ablations
                
                if Config.dataset == 'hopper-medium-expert-v2':
                    if avg_score_this < 0.53 and t>600:
                        return
                if Config.dataset == 'hopper-medium-v2':
                    if avg_score_this < 0.15 and t>400:
                        return
                    if avg_score_this < 0.32 and t>425:
                        return
                    if avg_score_this < 0.40 and t>500:
                        return
                if Config.dataset == 'hopper-medium-replay-v2':
                    if avg_score_this < 0.38 and t>500:
                        return
                    if avg_score_this < 0.53 and t>700:
                        return
                if Config.dataset == 'halfcheetah-medium-v2':
                    if avg_score_this < 0.065 and t>200:
                        return
                    if avg_score_this < 0.15 and t>400:
                        return
                    if avg_score_this < 0.27 and t>700:
                        return
                if Config.dataset == 'halfcheetah-medium-replay-v2':
                    if avg_score_this < 0.06 and t>200:
                        return
                    if avg_score_this < 0.15 and t>400:
                        return
                    if avg_score_this < 0.25 and t>800:
                        return
                if Config.dataset == 'walker2d-medium-replay-v2':
                    if avg_score_this < 0.035 and t>200:
                        return
                if Config.dataset == 'walker2d-medium-v2':
                    if avg_score_this < 0.33 and t>600:
                        return
            print('avg:', avg_score_this, 'time:', t, 'total done', total_done, path[40:])
            avg_scores.append(avg_score_this)
            times.append(t)
            stderr = np.std(normalized_scores) / np.sqrt(len(normalized_scores))
            std_errs.append(stderr)
        if t == 0 and os.path.isfile(path):
               print('WARNING: file already existed')
               print(path)
               return
        if t % 5 == 0 or t >= Config.max_path_length-2 :
            
            
            with open(path, mode='w') as file:
                writer = csv.writer(file)
                configuration_name = 'state_' + str(args.state_model_name) + '_action_' + str(args.action_model_name) + '_rew_' + str(args.reward_model_name)
                writer.writerow(['Average normalized score', 'Time', 'Std error','model name'])
                for i in range(len(avg_scores)):
                    writer.writerow([avg_scores[i], times[i], std_errs[i], configuration_name])




    episode_rewards = np.array(episode_rewards)
    normalized_scores = [env_list[0].get_normalized_score(s) for s in episode_rewards]
   
    print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}")
    print(normalized_scores, 'avg:', np.mean(normalized_scores))
    return normalized_scores
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
     
    parser.add_argument("--env_name", default="hopper-medium-v2", type=str)
    
    parser.add_argument("--reward_model_name", default="mlp", type=str)
    parser.add_argument("--action_model_name", default="transformer", type=str)
    parser.add_argument("--state_model_name", default="diffusion", type=str)

    parser.add_argument("--warmup", action='store_true', default=False)
    parser.add_argument("--inv_mlp", action='store_true', default=False)
    parser.add_argument("--statefolder", default="decision_plugins_state_transformer_pomdp", type=str)
    parser.add_argument("--stateckpt", default="state_1000000.pt", type=str)
    parser.add_argument("--actfolder", default="dtidm_acttransformer_withpositional", type=str)
    parser.add_argument("--actckpt", default="800000.pt", type=str)
    parser.add_argument("--rewfolder", default="orig_dg_pomdp_last2dim_rew", type=str)
    parser.add_argument("--rewckpt", default="state_1800000.pt", type=str)

    parser.add_argument("--conditional_w", type=float, default=1.8)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--testrt", type=float, default=0.8)
    parser.add_argument("--future_horizon", type=int, default=20)
    parser.add_argument("--obs_horizon", type=int, default=50)


    args = parser.parse_args()
    
    # Print the arguments
    print("env_name:", args.env_name)
    print("reward_model_name:", args.reward_model_name)
    print("action_model_name:", args.action_model_name)
    print("state_model_name:", args.state_model_name)
    print('-' * 10)
    print('use diffusion warm up:', args.warmup, '\n using inv mlp:', args.inv_mlp, '\n conditional_w:', args.conditional_w)
    print('noise scale:', args.noise_scale)
    print('-' * 10)
    Config.device = 'cuda'

    normalized_score = evaluate(args)

    


