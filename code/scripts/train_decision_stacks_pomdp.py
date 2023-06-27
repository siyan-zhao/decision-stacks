import diffuser.utils as utils
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from config.locomotion_config_decision_stacks_pomdp import Config
import os
import shutil
# three modules: state model, reward model and action model.
def main(args):
    
   

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#
    
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
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
    )


    torch.set_num_threads(1)
    dataset = dataset_config()
    
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#

    decision_plugins_config = utils.Config(
        'models.Decision_Stacks',
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
        reward_model_name=Config.reward_model_name,
        state_model_name=Config.state_model_name,
        action_model_name=Config.action_model_name,
        wor=args.wor
    )

    trainer_config = utils.Config(
        utils.TrainerThreemodule,
        savepath='trainer_config.pkl',
        train_batch_size=args.batch_size,
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

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    decision_plugins = decision_plugins_config()

    trainer = trainer_config(decision_plugins, dataset)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    batch = utils.batchify(dataset[0], Config.device)
    x, cond, returns, timesteps, rewards = batch
    
    loss, _, _, _ = decision_plugins.loss(*batch)
    loss.backward()

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)
    log_path = Config.bucket + '/'+ Config.dataset
    writer = SummaryWriter(log_path)
    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)
    
    for i in range(n_epochs):
        print('epoch:', i ,'/', n_epochs)
        trainer.train(n_train_steps=Config.n_steps_per_epoch, writer=writer)
    trainer.save()

if __name__ == '__main__':
      
    parser = argparse.ArgumentParser()
     
    parser.add_argument("--env_name", default="hopper-medium-v2", type=str)
    parser.add_argument("--reward_model_name", default="mlp", type=str)
    parser.add_argument("--action_model_name", default="mlp", type=str)
    parser.add_argument("--state_model_name", default="transformer", type=str)
    parser.add_argument("--wor", action='store_true', default=False) # used for ablation study
    parser.add_argument("--bucket", default='decision_plugins_state_transformer', type=str) # store path
    parser.add_argument('-traininv', "--train_inv", action='store_true')
    parser.add_argument('-trainstate', "--train_state", action='store_true')
    parser.add_argument('-trainrew', "--train_rew", action='store_true')
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()
    
    import json

    # Get the current file name and location
    current_file = os.path.abspath(__file__)
    # Set the destination directory for the file
    Config.bucket = '/home/data/ckpts/siyanz/dd_bayesian/' + args.bucket
    Config.dataset = args.env_name
    if 'hopper' in Config.dataset:
        Config.returns_scale = 400.0
    elif 'halfcheetah' in Config.dataset:
        Config.returns_scale = 1200.0
    elif 'walker' in Config.dataset:
        Config.returns_scale = 550.0 # Determined using rewards from the dataset

    
    Config.train_inv = args.train_inv
    Config.train_state = args.train_state
    Config.train_rew = args.train_rew
    Config.reward_model_name = args.reward_model_name
    Config.action_model_name = args.action_model_name
    Config.state_model_name = args.state_model_name

    log_path = Config.bucket + '/'+ args.env_name
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # Use shutil to copy the file to the destination directory
    shutil.copy2(current_file, log_path)
    # Print the arguments
    print("env_name:", args.env_name)
    print("train_inv:", args.train_inv)
    print("train_state:", args.train_state)
    print("train_rew:", args.train_rew)
    print("saved file:", args.bucket)
    print("reward_model_name:", args.reward_model_name)
    print("action_model_name:", args.action_model_name)
    print("state_model_name:", args.state_model_name)
    
    print('saving logs to:', log_path, '------------')
    with open(log_path + '/args.json', 'w') as f:
        json.dump(vars(args), f)
    print(Config.__dict__)
    
    with open(log_path + '/config.json', 'w') as f:
        json.dump(Config.__dict__, f, default=lambda x: str(x))

    main(args)