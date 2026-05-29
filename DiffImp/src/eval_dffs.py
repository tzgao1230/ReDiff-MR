# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : eval_dffs.py
# Time       ：12/11/2024 1:57 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description： 生成aug sequence
"""
import os
import torch
import einops
import tempfile
import numpy as np
import torch.nn.functional as F
from utils import set_random_seed
from utils import save_pickle
from diffuser import diff_utils as utils
from absl import logging
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from torch.utils.data import DataLoader
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from diffuser.diff_dataset import merge_miss
from utils import pad_list
import ipdb




def convert_input(batch_data, tokenizers, device, config_health=None):
    # padding到相同的长度便于转为tensor
    indices, batch = batch_data
    feature_keys = ['conditions', 'procedures', 'drugs']

    for feature_key in feature_keys:
        # tokenizers
        # print("KKKKKKKKK", batch[feature_key])
        batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])
        batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist'])  # B,V,M
        origin = batch[feature_key + '_comp'].copy()
        batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp'])  # B,M
        # tensor
        batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
        batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
        batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)
        # mask
        batch[feature_key + '_mask'] = batch[feature_key] != 0  # 为True的就是mask掉的, # B,M
        batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0  # 这个是为了padding, # B,V,M
        batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0  # 这个是为了padding # B,M

        # others
        batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(origin, padding=False,
                                                                                truncation=False)  # 不需要减去2,只需要能看清楚是什么就行，因为decode的时候会加上2
        batch[feature_key + '_comps_origin'] = origin  # origin ID

        batch['has_note'] = False
        if config_health['DATASET'] == 'MIV-Note':
            batch['note'] = torch.tensor(batch['note'], device=device)  # cur, B,D
            batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)  # hist, B,T,D
            batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) != 0  # B,T

            # 作为context,需要合并
            batch['note' + '_hist'] = torch.cat([batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1)  # B,T+1,D
            batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B,D
            batch['note' + '_hist_mask'] = torch.cat([batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)], dim=1)  # B,T+1

            batch['has_note'] = True


    batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)


    aligned_data = [batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for feature_key in
                    feature_keys]  # B,V,1
    batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B, V, 3

    return indices, batch



def evaluate_diff(best_model_path, config, config_health, hp_dataset, tokenizers, exp_num='0'):
    print("eval_diffusion")
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config["nnet_path"] = best_model_path

    set_random_seed(config_health['SEED'])
    data_n = config_health['DATASET']
    task = config_health['TASK']
    if config['sample'].get('scale', 0) != 0:
        logging.info('Pure conditional evaluation detected, forcing sample.scale=0 to disable CFG sampling.')
        config['sample']['scale'] = 0
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')

    utils.set_logger(log_level='info', fname=os.path.join(config['workdir'], data_n + '-' + task + '-' + exp_num + '-output.log'))

    dataset = get_dataset(data_n, aug=False, samples=hp_dataset, config=config)
    train_dataset = dataset.get_split(split='train', labeled=True)
    batch_size = config['train']['batch_size'] #// accelerator.num_processes
    num_train_visit = len(train_dataset)
    print("XXXXXXXXXXX", num_train_visit)

    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
                                     drop_last=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    nnet = utils.get_nnet(num_train_visit, tokenizers, **config['nnet'])
    logging.info(f'load nnet from {config["nnet_path"]}')
    nnet.load_state_dict(utils.load_trusted_torch_file(config['nnet_path'], map_location='cpu'))
    nnet = nnet.to(device)
    # nnet.test_e_step()

    nnet.eval()

    def select_sampling_context(context_cro, context_fin, timesteps, condition_mode='hybrid'):
        T = 1000
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=context_cro.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(context_cro.size(0))
        timesteps = timesteps.float()
        t_norm = timesteps / T

        switch_ratio = config.get('switch_ratio', 0.5)
        sigmoid_slope = config.get('sigmoid_slope', 10)

        c_norm = torch.nn.functional.layer_norm(context_cro, (context_cro.shape[-1],))
        f_norm = torch.nn.functional.layer_norm(context_fin, (context_fin.shape[-1],))

        if condition_mode == 'coarse_only':
            return c_norm
        if condition_mode == 'fine_only':
            return f_norm
        if condition_mode != 'hybrid':
            raise ValueError(f'Unsupported condition_mode: {condition_mode}')

        weight_injection = 1.0 - torch.sigmoid(sigmoid_slope * (t_norm - switch_ratio))
        if c_norm.dim() == 3:
            return c_norm + weight_injection[:, None, None] * (f_norm - c_norm)
        return c_norm + weight_injection[:, None] * (f_norm - c_norm)

    def cfg_nnet(x, timesteps, context, condition_mode='hybrid'):
        context_cro, context_fin = context

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=x.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x.size(0))
        timesteps = timesteps.float()
        selected_context = select_sampling_context(
            context_cro,
            context_fin,
            timesteps,
            condition_mode=condition_mode,
        )
        t_discrete = timesteps.long().clamp(0, 999)
        _cond = nnet(x, t_discrete, context=selected_context)
        # Pure conditional imputation keeps the sampling path aligned with training.
        # We preserve the previous CFG branch below as a commented ablation reference,
        # but do not activate it in the current route.
        #
        # if config['sample']['scale'] == 0:
        #     return _cond
        # _empty_context = torch.tensor(dataset.empty_context, device=device, dtype=selected_context.dtype)
        # _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        # _uncond = nnet(x, t_discrete, context=_empty_context)
        # return _uncond + config['sample']['scale'] * (_cond - _uncond)
        return _cond


    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
        return utils.decode_multiview_samples(
            _batch,
            get_logits,
            tokenizers,
            thred=thred,
            topk_per_view={'conditions': config.get('conditions_topk')}
        )


    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    test_generator = get_test_generator()
    _betas = utils.stable_diffusion_beta_schedule()
    N = len(_betas)

    logging.info(config['sample'])
    logging.info(f'sample: n_samples={len(test_dataset)}')

    def dpm_solver_sample(_n_samples, _sample_steps, condition_mode='hybrid', **kwargs):
        
        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device)  # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1, 1, config['z_shape'][1])  # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2)
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式
        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, kwargs['context'], condition_mode=condition_mode)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)  # embedding， 感觉这里需要rounding
        return decode(_z, nnet.get_logits)

    def sample_fn(_n_samples):
        indices, batch = next(test_generator)
        assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch), tokenizers, device, config_health)
        _z, context_cro, context_fin, _, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
        return labels, hard_labels, dpm_solver_sample(
            _n_samples,
            config['sample']['sample_steps'],
            condition_mode=config.get('sample_condition_mode', 'hybrid'),
            context=[context_cro, context_fin],
            z_init=_z,
            mask=sample_input['mask']
        )


    with tempfile.TemporaryDirectory() as temp_path:
        path = config['sample']['path'] + data_n + '/' + task or temp_path
        os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        utils.sample2dir(path, len(test_dataset), config['sample']['mini_batch_size'], sample_fn,tokenizers,
                         dataset.unpreprocess)
        _quality, _jaccard = utils.calculate_quality(path)
        logging.info(f'nnet_path={config["nnet_path"]}, quality={_quality}, jaccard={_jaccard}') # 日志不会少，但是tem_path会少


def load_diffusion(config, tokenizers):
    # data_n = config_health['DATASET']
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')
    # dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=config)
    # train_dataset = dataset.get_split(split='train', labeled=True)
    # batch_size = config['train']['batch_size'] #// accelerator.num_processes
    nnet = utils.get_nnet(config['num_train_visit'], tokenizers, **config['nnet'])
    logging.info(f'load nnet from {config["nnet_path"]}')  # 这个path哪里来的
    nnet.load_state_dict(utils.load_trusted_torch_file(config['nnet_path'], map_location='cpu'))
    nnet = nnet.to(device)
    return nnet



def augment_dataset(nnet, config, config_health, hp_dataset, tokenizers, root_to=None):
    nnet.eval()
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')
    # 转换数据集
    data_n = config_health['DATASET']
    task = config_health['TASK']
    if config['sample'].get('scale', 0) != 0:
        logging.info('Pure conditional augmentation detected, forcing sample.scale=0 to disable CFG sampling.')
        config['sample']['scale'] = 0
    # 下面开始补全
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=config) # 无关。
    batch_size = config['train']['batch_size']  # // accelerator.num_processes
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True,
                                      collate_fn=collate_fn_dict)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=False,
                                     drop_last=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True,
                                     collate_fn=collate_fn_dict)

    # def cfg_nnet(x, timesteps, context):
    #     # classifier-free
    #     _cond = nnet(x, timesteps, context=context)
    #     if config['sample']['scale'] == 0:
    #         return _cond
    #     # ipdb.set_trace()
    #     _empty_context = torch.tensor(dataset.empty_context, device=device, dtype=selected_context.dtype)
    #     _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        
    #     _uncond = nnet(x, timesteps, context=_empty_context)
    #     return _cond + config['sample']['scale'] * (_cond - _uncond)
    def select_sampling_context(context_cro, context_fin, timesteps, condition_mode='hybrid'):
        T = 1000
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=context_cro.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(context_cro.size(0))
        timesteps = timesteps.float()
        t_norm = timesteps / T

        switch_ratio = config.get('switch_ratio', 0.5)
        sigmoid_slope = config.get('sigmoid_slope', 10)

        c_norm = torch.nn.functional.layer_norm(context_cro, (context_cro.shape[-1],))
        f_norm = torch.nn.functional.layer_norm(context_fin, (context_fin.shape[-1],))

        if condition_mode == 'coarse_only':
            return c_norm
        if condition_mode == 'fine_only':
            return f_norm
        if condition_mode != 'hybrid':
            raise ValueError(f'Unsupported condition_mode: {condition_mode}')

        weight_injection = 1.0 - torch.sigmoid(sigmoid_slope * (t_norm - switch_ratio))
        if c_norm.dim() == 3:
            return c_norm + weight_injection[:, None, None] * (f_norm - c_norm)
        return c_norm + weight_injection[:, None] * (f_norm - c_norm)

    def cfg_nnet(x, timesteps, context, condition_mode='hybrid'):
        """
        与 train_dffs.py 中 dpm_solver_sample 的 coarse->fine 融合保持一致：
        1) LayerNorm(context)
        2) weight_injection = 1 - sigmoid(slope * (t_norm - switch_ratio))
        3) context = c_norm + weight_injection * (f_norm - c_norm)
        """
        context_cro, context_fin = context

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=x.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x.size(0))
        timesteps = timesteps.float()
        selected_context = select_sampling_context(
            context_cro,
            context_fin,
            timesteps,
            condition_mode=condition_mode,
        )
        t_discrete = timesteps.long().clamp(0, 999)
        _cond = nnet(x, t_discrete, context=selected_context)

        # 5. Pure conditional augmentation: keep inference aligned with the
        # pure conditional training route. Old CFG code is kept as comments above
        # for ablation reference, but is not used here.
        return _cond

    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
        return utils.decode_multiview_samples(
            _batch,
            get_logits,
            tokenizers,
            thred=thred,
            topk_per_view={'conditions': config.get('conditions_topk')}
        )

    def get_train_generator():
        while True:
            for indices, data in train_dataset_loader:
                yield indices, data

    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    train_generator = get_train_generator()
    test_generator = get_test_generator()


    _betas = utils.stable_diffusion_beta_schedule()
    N = len(_betas)

    def dpm_solver_sample(_n_samples, _sample_steps, condition_mode='hybrid', **kwargs):
        """去噪声"""

        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device)  # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1, 1, config['z_shape'][1])  # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2)  # 从正态分布中采样噪声。 # B,3,128
        # 先加噪声，重新定义schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式
        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, kwargs['context'], condition_mode=condition_mode)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)  # embedding， 感觉这里需要rounding

        return decode(_z, nnet.get_logits)

    def sample_fn_train(_n_samples):
        indices, batch = next(train_generator)
        # print("CCCBBB", indices.size(0), _n_samples)
        assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch), tokenizers, device, config_health)
        # _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
        _z, context_cro, context_fin, _, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
        return labels, hard_labels, dpm_solver_sample(
            _n_samples,
            config['sample']['sample_steps'],
            condition_mode=config.get('sample_condition_mode', 'hybrid'),
            context=[context_cro, context_fin],
            z_init=_z,
            mask=sample_input['mask']
        )

    def sample_fn(_n_samples):
        indices, batch = next(test_generator)
        assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch),tokenizers, device, config_health)
        # _z, _context_emb, _, labels,hard_labels = nnet.encode_everything_test(indices, **batch)
        # return labels,hard_labels, dpm_solver_sample(_n_samples, config['sample']['sample_steps'], context=_context_emb,  z_init=_z, mask=sample_input['mask'])
        _z, context_cro, context_fin, _, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
        return labels, hard_labels, dpm_solver_sample(
            _n_samples,
            config['sample']['sample_steps'],
            condition_mode=config.get('sample_condition_mode', 'hybrid'),
            context=[context_cro, context_fin],
            z_init=_z,
            mask=sample_input['mask']
        )


    logging.info(config['sample'])

    # for train
    logging.info(f'sample: n_samples={len(train_dataset)}')
    with tempfile.TemporaryDirectory() as temp_path:
        path = config['sample']['path'] + data_n + '/' + task or temp_path
        os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        aug_train_sample, aug_train_hardsample = utils.sample2dir_test(path, len(train_dataset), config['sample']['mini_batch_size'], sample_fn_train,
                         dataset.unpreprocess)

    # for test
    logging.info(f'sample: n_samples={len(test_dataset)}')
    with tempfile.TemporaryDirectory() as temp_path:
        path = config['sample']['path'] + data_n + '/' + task or temp_path
        os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        aug_test_sample, aug_test_hardsample = utils.sample2dir_test(path, len(test_dataset), config['sample']['mini_batch_size'], sample_fn,
                         dataset.unpreprocess)

    if config_health['MODEL'] == 'ours':
        soft = False
        aug_train_sample = merge_miss(hp_dataset[0], aug_train_hardsample, soft=soft)
        aug_test_sample = merge_miss(hp_dataset[1], aug_test_hardsample, soft=soft)
    elif config_health['MODEL'] == 'MedDiff':
        soft = True
        aug_train_sample = merge_miss(hp_dataset[0], aug_train_sample, soft=soft)
        aug_test_sample = merge_miss(hp_dataset[1], aug_test_sample, soft=soft)


    # 应当把D*3*d放到原有的dataset里面
    if root_to is not None:
        # 存储以便读取
        save_pickle(aug_train_sample, root_to+'train_samples.pkl')
        save_pickle(aug_test_sample, root_to+'test_samples.pkl')
        print("All aug data save to {}.".format(path))

    return aug_train_sample, aug_test_sample
