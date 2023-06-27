
CUDA_VISIBLE_DEVICES=4 python train_decision_stacks_mdp.py \
                            --env_name 'halfcheetah-medium-replay-v2' \
                            --bucket 'ds_acttrans_mdp_eval' \
                            --action_model_name transformer \
                            -traininv \
                            --batch_size 32

# CUDA_VISIBLE_DEVICES=5 python train_decision_stacks_mdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_actmlp_mdp' \
#                             --action_model_name mlp \
#                             -traininv \
#                             --batch_size 32

# CUDA_VISIBLE_DEVICES=4 python train_decision_stacks_pomdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_acttrans_pomdp_eval' \
#                             --action_model_name transformer \
#                             -traininv \
#                             --batch_size 32

# CUDA_VISIBLE_DEVICES=4 python train_decision_stacks_mdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_actdiffusion_mdp_eval' \
#                             --action_model_name diffusion \
#                             -traininv \
#                             --batch_size 32

# CUDA_VISIBLE_DEVICES=5 python train_decision_stacks_pomdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_actdiffusion_pomdp_final' \
#                             --action_model_name diffusion \
#                             -traininv \
#                             --batch_size 32


