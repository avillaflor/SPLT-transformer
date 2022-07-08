from src.utils import watch

#------------------------ base ------------------------#

logbase = 'logs/'
gpt_expname = 'gpt'

## automatically make experiment names for planning
## by labelling folders with these args
args_to_watch = [
    ('prefix', ''),
    ('max_context_transitions', 'mx'),
    ('horizon', 'H'),
]

base = {
    'train': {
        'discount': 0.99,
        'n_layer': 4,
        'n_head': 8,
        'n_embd': 16,

        ## number of epochs for a 1M-size dataset; n_epochs = 1M / dataset_size * n_epochs_ref
        'n_epochs_ref': 50,
        'logbase': logbase,
        'device': 'cuda',

        'batch_size': 256,
        'learning_rate': 1e-4,
        'lr_decay': False,
        'seed': 0,

        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,

        'step': 1,
        'subsampled_sequence_length': 10,
        'termination_penalty': None,
        'exp_name': gpt_expname,

        'action_weight': 5,
        'reward_weight': 1,
        'value_weight': 1,
    },

    'plan': {
        'logbase': logbase,
        'gpt_loadpath': gpt_expname,
        'gpt_epoch': 'latest',
        'device': 'cuda',

        'plan_freq': 1,
        'horizon': 5,

        'max_context_transitions': 5,
        'prefix_context': True,

        'exp_name': watch(args_to_watch),
        'prefix': 'plans/defaults/',
        'suffix': '0',
        'verbose': True,
        'seed': 0,
    },

    'bc_train': {
        'discount': 1.0,
        'n_layer': 4,
        'n_head': 8,
        'n_embd': 16,

        ## number of epochs for a 1M-size dataset; n_epochs = 1M / dataset_size * n_epochs_ref
        'n_epochs_ref': 50,
        'logbase': logbase,
        'device': 'cuda',

        'batch_size': 256,
        'learning_rate': 1e-4,
        'lr_decay': False,
        'seed': 0,

        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,

        'step': 1,
        'subsampled_sequence_length': 10,
        'termination_penalty': None,
        'exp_name': gpt_expname,

        'action_weight': 5,
        'reward_weight': 1,
        'value_weight': 1,
    },

    'bc_plan': {
        'logbase': logbase,
        'gpt_loadpath': gpt_expname,
        'gpt_epoch': 'latest',
        'device': 'cuda',

        'plan_freq': 1,
        'max_context_transitions': 5,
        'prefix_context': True,

        'exp_name': watch(args_to_watch),
        'prefix': 'plans/defaults/',
        'suffix': '0',
        'verbose': True,
        'seed': 0,
    },


    'tt_train': {
        'N': 100,
        'discount': 0.99,
        'n_layer': 4,
        'n_head': 8,
        'n_embd': 16,

        ## number of epochs for a 1M-size dataset; n_epochs = 1M / dataset_size * n_epochs_ref
        'n_epochs_ref': 50,
        'logbase': logbase,
        'device': 'cuda',

        'batch_size': 256,
        'learning_rate': 6e-4,
        'lr_decay': True,
        'seed': 0,

        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,

        'step': 1,
        'subsampled_sequence_length': 10,
        'termination_penalty': None,
        'exp_name': gpt_expname,

        'discretizer': 'QuantileDiscretizer',
        'action_weight': 5,
        'reward_weight': 1,
        'value_weight': 1,
    },

    'tt_plan': {
        'logbase': logbase,
        'gpt_loadpath': gpt_expname,
        'gpt_epoch': 'latest',
        'device': 'cuda',

        'plan_freq': 1,
        'horizon': 5,
        'beam_width': 128,
        'n_expand': 2,

        'k_obs': 1,
        'k_act': None,
        'cdf_obs': None,
        'cdf_act': 0.6,
        'percentile': 'mean',

        'max_context_transitions': 5,
        'prefix_context': True,

        'exp_name': watch(args_to_watch),
        'prefix': 'plans/defaults/',
        'suffix': '0',
        'verbose': True,
        'seed': 0,
    },

}


#------------------------ toy car ------------------------#
idm_uniform07 = {
    'train': {
        'n_epochs_ref': 5.0,
    },
    'plan': {},
    'bc_train': {
        'n_epochs_ref': 5.0,
    },
    'bc_plan': {},
    'tt_train': {
        'n_epochs_ref': 5.0,
    },
    'tt_plan': {
        'horizon': 5,
        'k_obs': 1,
        'cdf_obs': None,
        'cdf_act': 0.6,
    },
    # More aggressive planning
    #  'tt_plan': {
        #  'horizon': 5,
        #  'k_obs': 1,
        #  #  'k_obs': 5,
        #  'k_act': 20,
        #  'cdf_obs': None,
        #  'cdf_act': None,
    #  }
}


#------------------------ locomotion ------------------------#
halfcheetah_medium_v2 = halfcheetah_medium_replay_v2 = halfcheetah_medium_expert_v2 = \
        walker2d_medium_v2 = walker2d_medium_replay_v2 = walker2d_medium_expert_v2 = \
        hopper_medium_v2 = hopper_medium_replay_v2 = hopper_medium_expert_v2 = {
    'train': {
        'n_epochs_ref': 25,
    },
    'plan': {
        'horizon': 10,
    },
}

#------------------------ carla ------------------------#
random_ttc = {
    'train': {},
    'plan': {
        'horizon': 5,
        'max_context_transitions': 2,
    },
    'tt_train': {},
    'tt_plan': {},
    'bc_train': {
        'subsampled_sequence_length': 15,
    },
    'bc_plan': {}
}
