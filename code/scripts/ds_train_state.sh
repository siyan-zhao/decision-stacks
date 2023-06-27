
# CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_pomdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_statetrans_pomdp_eval' \
#                             --state_model_name transformer \
#                             -trainstate \
#                             --batch_size 32

# CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_mdp.py \
#                             --env_name 'halfcheetah-medium-replay-v2' \
#                             --bucket 'ds_statediff_mdp_eval' \
#                             --state_model_name diffusion \
#                             -trainstate \
#                             --batch_size 32

CUDA_VISIBLE_DEVICES=3 python train_decision_stacks_pomdp.py \
                            --env_name 'halfcheetah-medium-replay-v2' \
                            --bucket 'ds_statediff_pomdp_eval' \
                            --state_model_name diffusion \
                            -trainstate \
                            --batch_size 32