
import os

eval_testrt = { 
            'hopper-medium-replay-v2': 0.85, 
            'walker2d-medium-replay-v2': 0.65, 
            'halfcheetah-medium-replay-v2': 0.4, 
            'walker2d-medium-v2': 0.75, 
            'halfcheetah-medium-v2': 0.5, 
            'hopper-medium-v2': 0.85, 
            'hopper-medium-expert-v2': 0.85, 
            'walker2d-medium-expert-v2': 0.9, 
            'halfcheetah-medium-expert-v2': 0.8, 
        }

def evaluate_if_trained():
   
    env = 'halfcheetah-medium-replay-v2'
    cmd = " python eval_pomdp_decision_stacks.py \
        --env_name {} \
        --stateckpt state_state_best.pt \
        --state_model_name diffusion \
        --statefolder ds_statediff_pomdp_eval \
        --action_model_name transformer \
        --actfolder ds_acttrans_pomdp_eval \
        --actckpt best.pt \
        --reward_model_name transformer \
        --rewfolder ds_rewtrans_pomdp_eval \
        --rewckpt best.pt \
        --testrt {} \
        --noise_scale 0.5 \
        --conditional_w 2.2 \
        --future_horizon 20 \
        --obs_horizon 50 \
        ".format(env, eval_testrt[env])
    try:
        os.system(cmd)
    except:
        pass


evaluate_if_trained()


