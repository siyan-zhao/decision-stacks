import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize
    bucket = ''
    dataset = ''
    reward_model_name = ''
    action_model_name = ''
    state_model_name = ''
    train_inv = False
    train_rew = False
    train_state = False
    
    ## model
    model = 'models.TemporalUnet'
    decision_model = 'models.Decision_Stacks'
    horizon = 100
    n_diffusion_steps = 200
    transformer_rew = True
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w=1.5
    test_ret=0.9
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.SequenceDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 1000
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = -100
    if 'hopper' in dataset:
        returns_scale = 400.0
    elif 'halfcheetah' in dataset:
        returns_scale = 1200.0
    elif 'walker' in dataset:
        returns_scale = 550.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 10000
    loss_type = 'l2'
    n_train_steps = 2e6
    batch_size = 32
    learning_rate = 5e-4
    gradient_accumulate_every = 1
    ema_decay = 0.995
    log_freq = 1000
    save_freq = 100000
    sample_freq = 10000
    n_saves = 5
    save_parallel = False
    n_reference = 8
    save_checkpoints = True