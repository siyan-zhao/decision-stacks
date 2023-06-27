
# CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_mdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_rewtrans_mdp_eval' \
#                             --reward_model_name transformer \
#                             -trainrew \
#                             --batch_size 32



# CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_pomdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_rewtrans_pomdp_eval' \
#                             --reward_model_name transformer \
#                             -trainrew \
#                             --batch_size 32

# CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_mdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_rewdiff_mdp_eval_final' \
#                             --reward_model_name diffusion \
#                             -trainrew \
#                             --batch_size 32

CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_mdp.py \
                            --env_name 'halfcheetah-medium-replay-v2' \
                            --bucket 'ds_rewmlp_mdp_eval' \
                            --reward_model_name mlp \
                            -trainrew \
                            --batch_size 32

# CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_pomdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_rewmlp_pomdp_eval_final2' \
#                             --reward_model_name mlp \
#                             -trainrew \
#                             --batch_size 32

# CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_pomdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_rewdiff_pomdp_eval_final' \
#                             --reward_model_name diffusion \
#                             -trainrew \
#                             --batch_size 32