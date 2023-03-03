import torch
import os

from model.model import Model

args = {
    'model_list': {
        'Model': Model
    },
    'pretrain': True,
    'platform': 'd1080t',
    'n_epochs': 10,
    'learning_rate': 1e-5,
    'batch_size': 2,
    'record': 10,

    'init_h': 180,
    'init_w': 360,
    'patch_size': 30,
    'ratio': 0.6,
    'sequence_length': 6,
    'lead_time': 19,
    'dropout': 0,

    'nc_file': f'./file/HadISST_sst.nc',
    'raw_file': f'./file/min_max_sst.npz',
    'train_npz_file': r'./file/train.npz',
    'forecast_npz_file': r'./file/forecast.npz',

    '3090': {
        'encoder_growth_rate': 12,
        'encoder_bn_size': 2,
        'encoder_theta': 0.5,
        'encoder_dense_config': [2, 4],
        'encoder_pool_size': [2, 3],

        'overlap': True,
        'rank': 8,
        'k': 0.6,
        'head': 4,
        'concat': False,
        'add_self_loops': True,
        'samples': 5,
        'num_layers': 2,

        'decoder_reduce_rate': 3,
        'decoder_bn_size': 2,
        'decoder_theta': 0.5,
        'decoder_reduce_config': [2, 4],
        'decoder_pool_size': [3, 2]
    },

    '1060': {
        'encoder_growth_rate': 12,
        'encoder_bn_size': 2,
        'encoder_theta': 0.5,
        'encoder_dense_config': [2, 4],
        'encoder_pool_size': [2, 3],

        'overlap': True,
        'rank': 8,
        'k': 0.6,
        'head': 2,
        'concat': False,
        'add_self_loops': True,
        'samples': 5,
        'num_layers': 2,

        'decoder_reduce_rate': 3,
        'decoder_bn_size': 2,
        'decoder_theta': 0.5,
        'decoder_reduce_config': [2, 4],
        'decoder_pool_size': [3, 2]
    },

    'd1080t': {
        'encoder_growth_rate': 16,
        'encoder_bn_size': 2,
        'encoder_theta': 0.5,
        'encoder_dense_config': [5, 6],
        'encoder_pool_size': [2, 3],

        'overlap': True,
        'rank': 8,
        'k': 0.6,
        'head': 4,
        'concat': False,
        'add_self_loops': True,
        'samples': 25,
        'num_layers': 2,

        'decoder_reduce_rate': 3,
        'decoder_bn_size': 2,
        'decoder_theta': 0.5,
        'decoder_reduce_config': [5, 6],
        'decoder_pool_size': [3, 2]
    }
}

if __name__ == '__main__':
    print(args['3090'])
