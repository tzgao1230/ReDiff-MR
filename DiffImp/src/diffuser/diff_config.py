# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : diff_config.py
# Time       ：8/11/2024 5:12 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：
"""

class DIFFCONFIG(): # 不要有drugs
    """DRL config"""
    seed = 528
    device = 6
    warm = 10
    benchmark = False

    ckpt_root = '/home/qluai/tianzhi1/MMHealth/log/difflog/ckpt/'
    sample_dir = '/home/qluai/tianzhi1/MMHealth/log/difflog/sample/'
    workdir = '/home/qluai/tianzhi1/MMHealth/difflog/work_dir/' # wandb
    output_path = '/home/qluai/tianzhi1/MMHealth/log/difflog/work_dir/output.log'
    z_shape = (3, 128)
    train_val_num = 50
    dim=128
    threshod = 0.4 # jaccard
    conditions_topk = 20
    switch_ratio = 0.3
    sigmoid_slope = 10
    fine_dropout = 0.3
    model_select_mse_quantile = 0.2
    model_select_mse_guard_ratio = 1.05
    early_stop_patience = 4
    early_stop_min_delta = 0.2
    force_restart = False


    dataset = {
        'MIII':{
            'name': 'MIII',
            'feature_keys':  ['conditions', 'procedures', 'drugs'],
            'cfg' : True,
            # Pure conditional imputation: disable CFG dropout in training.
            'p_uncond' : 0.0,
        },
        'MIV':{
            'name': 'MIV',
            'feature_keys':  ['conditions', 'procedures', 'drugs'],
            'cfg' : True,
            'p_uncond' : 0.0,
        },
        'MIV-Note': {
            'name': 'MIV-Note',
            'feature_keys': ['conditions', 'procedures', 'drugs'],
            'cfg': True,
            'p_uncond': 0.0,
        },
        'eICU': {
            'name': 'eICU',
            'feature_keys': ['conditions', 'procedures', 'drugs'],
            'cfg': True,
            'p_uncond': 0.0,
        }
    }
    sample = {
        'mini_batch_size': 16, # 训练的时候搞大点似乎也行； 但是eCIU不行，数量太少了，一次不够；
        # Pure conditional sampling: disable classifier-free guidance.
        'scale': 0,
        'path': '/home/qluai/tianzhi1/MMHealth/log/difflog/eval_sample/',
        'sample_steps': 20, # dpm solver用更少的步
        'n_samples': 10000,
    }
    hparams = 'lr' # monitor hyper

    nnet = {
        'name':'dit',
        'visit_config':  {
            'dim': 128,
            'num_heads': 2,
            'num_ids':2,
            'mode': 'sum',
            'logits_mode':'linear'
        },
        'con_config':{
            'dim': 128,
            'mode': 'concat',# joint concat
            'num_heads': 2,
            'hist_mode': 'individual', # joint  individual
        },
        'cluster_config':{
            'dim': 128,
            'k': 10, # 5， 10， 20，100
            'proto_reg': 1e-4,
            'ssl_temp': 0.1,
            'mode': 'k-means',
        },
        # 'img_size': 32,
        'in_chans' : 4,
        # 'patch_size' :  2,
        'embed_dim' :  128,
        'depth' :  2, # 12
        'num_heads' :  4,
        'mlp_ratio' :  4,
        'qkv_bias' :  False,
        'mlp_time_embed' :  False,
        'contx_dim' : 128,
        'num_contx_token' :  6 # 可能要改, 10 for note
            }


    optimizer = {
        'name': 'adamw',
        'lr' : 5e-4, #0.0002,eicu 用5e-3       # tianzhi add 5e-4
        'weight_decay' : 0,#0.03,
        'betas' : (0.9, 0.9),
    }

    lr_scheduler = {
        'name': 'warmup_cosine',
        'warmup_steps': 3000,
        'min_lr_scale': 0.1,
    }
    
    train = {
        'n_epochs': 100, # step/ epcoh
        'batch_size': 16,
        'log_interval': 10000, # 就直接存
        'eval_epoch': 2, # 更频繁地观察 MSE 低谷，避免错过中期最优 checkpoint
        # 'save_interval': 1000,
        # 'n_steps': 5000 # 这里的值 wrong
    }





  


config = vars(DIFFCONFIG)
config = {k:v for k,v in config.items() if not k.startswith('__')}
