# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：4/11/2024 3:27 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：nohup python main_unify.py > tes_ours_miv_phe.log 2>&1 &
"""

MIII_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '0',
    'EPOCH': 50, # 20 for rebuttal
    'WARM_EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 32,
    'DROPOUT': 0.1,
    'WD': 0.,#1e-3,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX' : 0.00001,

}

MIV_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '0',
    'EPOCH': 50,
    'WARM_EPOCH': 10,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 32,
    'DROPOUT': 0.2, # 0. # PHE
    'WD': 5e-4, #0.
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX': 0.0001,

}

##################################################
eICU_PARAMS = {
    'FEATURE': ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '0',
    'EPOCH': 50,
    'WARM_EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3, #
    'BATCH': 32,
    'DROPOUT': 0.3,
    'WD': 5e-4,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX': 0.001,

}

MIV_Note_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '0',
    'EPOCH': 50,
    'WARM_EPOCH': 10,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 32,
    'DROPOUT': 0.3, # 0. # PHE
    'WD': 5e-4, #0.
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX': 0.001,

}

class PHECONFIG(): # 不要有drugs
    """DRL config"""
    # data_info parameter
    DEV = False
    MODEL = "ours"
    PLM = "Clinical-BERT"
    TASK = 'PHE'
    DATASET = 'MIII'
    LABEL = 'labels'

    ATCLEVEL = 3
    RATIO = 0.6 # train-test split
    THRES = 0.4 # pred threshold
    # train parameter

    DATASET_PARAMS = {
        'MIII': MIII_PARAMS,
        'eICU': eICU_PARAMS,
        'MIV-Note': MIV_Note_PARAMS,
        'MIV' : MIV_PARAMS
        # 'EHR-SHOT': EHR_PARAMS,
        # 'OMOP': OMOP_PARAMS,
    }

    @classmethod
    def get_params(cls):
        return cls.DATASET_PARAMS.get(cls.DATASET, {})

    # log
    LOGDIR = '/home/qluai/tianzhi1/MMHealth/log/ckpt/'



# config = vars(PHECONFIG)
# config = {k:v for k,v in config.items() if not k.startswith('__')}

config = {**vars(PHECONFIG), **PHECONFIG.get_params()}
config = {k: v for k, v in config.items() if not k.startswith('__')}
