# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_dffs.py
# Time       ：11/11/2024 4:05 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：仿照CV重建
"""

import os
import torch
import tempfile
import einops
import wandb
import torch.nn.functional as F
import numpy as np
from absl import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from diffuser import diff_utils as utils
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from utils import set_random_seed
from utils import pad_list
import ipdb 



def run_diffusion(config, config_health, hp_dataset, tokenizers, exp_num='0'):
    """
    :param config: diffusion config
    :param config_health: healthcare config
    :param hp_dataset: healthcare train dataset
    :param tokenizers: healthcare tokenizers
    :return:
    """
    print("run_diffusion")
    set_random_seed(config_health['SEED'])

    data_n = config_health['DATASET']
    task = config_health['TASK']
    p_uncond = config['dataset'][data_n].get('p_uncond', 0.0)
    if p_uncond <= 0 and config['sample'].get('scale', 0) != 0:
        logging.info('Pure conditional training detected, forcing sample.scale=0 to disable CFG sampling.')
        config['sample']['scale'] = 0

    # utils.calculate_quality(config['sample']['path'] + data_n)

    # 加速
    # accelerator = accelerate.Accelerator() # CUDA_VISIBLE_DEVICES=4  accelerate launch main.py
    # device = accelerator.device
    # accelerate.utils.set_seed(config['seed'], device_specific=True)
    # device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')

    # logging.info(f'Process {accelerator.process_index} using device: {device}')
    logging.info(f'using device: {device}')

    # config['mixed_precision'] = accelerator.mixed_precision  # 混合精度
    # assert config['train']['batch_size'] % accelerator.num_processes == 0
    batch_size = config['train']['batch_size'] #// accelerator.num_processes

    # if accelerator.is_main_process:  # 主进程用作模型加载，记录日志等功能
    os.makedirs(config['ckpt_root'] + data_n + '/' + task, exist_ok=True)
    os.makedirs(config['sample_dir'] + data_n + '/' + task, exist_ok=True)

    # accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    wandb.init(dir=os.path.abspath(config['workdir']), project=f'mmht_{data_n}_{task}', config=config,
               name=config['hparams'], job_type='train', mode='online')  # 离线存储
    utils.set_logger(log_level='info', fname=os.path.join(config['workdir'], data_n +'-'+ task +'-' + exp_num + '-output.log'))
    logging.info(config)
    # else:
    #     utils.set_logger(log_level='error')  # 禁用打印
    #     builtins.print = lambda *args: None
    # logging.info(f'Run on {accelerator.num_processes} devices')

    # dataset
    dataset = get_dataset(data_n, aug=False, samples=hp_dataset, config=config) # training 所以aug设置为False
    # assert os.path.exists(dataset.fid_stat)  # fid起到啥作用？label的作用
    
    train_dataset = dataset.get_split(split='train', labeled=True)          # tianzhi add 这里是对原来划分的train_dataset处理后的数据集，具体转化有点复杂
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
                                     drop_last=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)

    print("Input for diffusion", train_dataset[0])      # 注意这里的数据长什么样子
    print("AAAAAAAA", (len(train_dataset)//batch_size +1)*batch_size, len(train_dataset)) # 如果dropout为False，就不要+1

    # model define
    train_state = utils.initialize_train_state(config, device, tokenizers, len(train_dataset)) # 多一个batch
    # nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
    #     train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader
    lr_scheduler = train_state.lr_scheduler

    if config.get('force_restart', False):
        logging.info('force_restart=True, skip resuming diffusion checkpoint')
    else:
        train_state.resume(config['ckpt_root'] + data_n + '/' + task)  # load latest ckpt

    def sync_cluster_state(source_model, target_model):
        with torch.no_grad():
            target_model.update_train_visit_emb(source_model.total_visit_emb)
            if hasattr(source_model, 'visit_2cluster'):
                target_model.visit_2cluster = source_model.visit_2cluster.detach().clone()
            target_model.update_centroid_emb(source_model.visit_centroids)

    # @ torch.cuda.amp.autocast() # 半精度

    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
        return utils.decode_multiview_samples(
            _batch,
            get_logits,
            tokenizers,
            thred=thred,
            topk_per_view={'conditions': config.get('conditions_topk')}
        )

    def get_data_generator():
        while True:
            for indices, data in tqdm(train_dataset_loader, desc='epoch'): # disable=not accelerator.is_main_process
                yield indices, data

    data_generator = get_data_generator()  # 转为生成器

    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    test_generator = get_test_generator()

    # diffusion prepare
    _betas = utils.stable_diffusion_beta_schedule()              # 原始调度方式，线性平方调度
    # _betas =  utils.cosine_beta_schedule_v2()
    _schedule = utils.Schedule(_betas) # forward add noise schedule
    logging.info(f'use {_schedule}')

    def cfg_nnet(x, timesteps, context):
        # Pure conditional imputation uses scale=0, so this function falls back to the
        # conditional branch directly. We keep the CFG path commented in behavior via config
        # for future ablations, but the current training run does not rely on it.
        _cond = nnet_ema(x, timesteps, context=context)
        if config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device, dtype=context.dtype)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, timesteps, context=_empty_context)
        guidance_scale = config['sample']['scale']
        return _uncond + guidance_scale * (_cond - _uncond)

    def select_sampling_context(context_cro, context_fin, timesteps, condition_mode='hybrid'):
        """Keep sampling-time context selection aligned with training-time coarse->fine mixing."""
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
    # tianzhi add 原版train
    # def train_step(indices, _batch, epoch=None):
    #     _metrics = dict()
    #     optimizer.zero_grad()
    #     # _z, _z_mask, _z_his = encode(_batch[0], _batch[1], batch[2]) # B, 3, D; B, 3, D
    #     # _zc, loss_c = cluster(_z) # B, 1, D, near cluster
    #     _z, _context, neigh_loss, _, _ = nnet.encode_everything(indices, **_batch) # B，3，D； B,9,D； 对比损失

    #     loss_mse, loss_round, T_loss = utils.LSimple(_z, nnet, _schedule,
    #                         context=_context,  **_batch)  # currently only support the extracted feature version
    #     _metrics['loss_mse'] = loss_mse.detach().mean() #accelerator.gather(loss.detach()).mean()
    #     _metrics['loss_c'] = neigh_loss.detach().mean() #accelerator.gather(neigh_loss.detach()).mean()
    #     _metrics['loss_rounding'] = loss_round.detach().mean() # batch
    #     _metrics['T_loss'] = T_loss.detach().mean() # batch
    #     total_loss = loss_mse.mean() + neigh_loss.mean() + loss_round.mean() + T_loss.mean() if epoch > config['warm'] else loss_mse.mean() + loss_round.mean() + T_loss.mean()
    #     total_loss.backward()
    #     # accelerator.backward(loss.mean() + neigh_loss.mean())

    #     optimizer.step()
    #     lr_scheduler.step()
    #     train_state.ema_update(config.get('ema_rate', 0.9999))
    #     train_state.step += 1
    #     return _z, dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    # tianzhi add 用于mimic3药物推荐的train_step 20260327
    # 监控转换时机
    def train_step(indices, _batch, epoch=None):
        _metrics = dict()
        optimizer.zero_grad()

        # ============================================================
        # [Tianzhi Add] Fine Context Dropout (核心修改)
        # 目的：以 30%~50% 的概率随机抹除患者的历史记录 (Fine Context)
        # 强迫模型在没有历史细节时，学会依赖 Coarse Context (原型)
        # ============================================================
        # 1. 设置 dropout 概率 (建议 0.3 到 0.5)
        fine_dropout_prob = config.get('fine_dropout', 0.3) 
        # 2. 生成随机掩码 (True 表示要丢弃历史)
        # indices.shape[0] 是 batch_size
        if fine_dropout_prob > 0:
            # 生成一个形状为 [B] 的布尔向量
            drop_mask = torch.rand(indices.shape[0], device=device) < fine_dropout_prob
            
            # 如果这一个 batch 里有需要 drop 的样本
            if drop_mask.any():
                # 遍历三个特征视图
                feature_keys = ['conditions', 'procedures', 'drugs']
                for key in feature_keys:
                    hist_mask_key = key + '_hist_mask'
                    if hist_mask_key in _batch:
                        # 将选中的样本的历史掩码全部置为 False (即视为无历史记录)
                        # _batch[hist_mask_key] shape: [B, V, M]
                        # drop_mask shape: [B] -> 广播到 [B, V, M]
                        _batch[hist_mask_key][drop_mask] = False

        # ============================================================

        # 1. 特征编码
        _z, context_cro, context_fin, neigh_loss, cluster_metrics, _, _ = nnet.encode_everything(indices, **_batch)

        # Pure conditional training: do not drop conditioning information to build
        # an unconditional branch. For imputation, preserving condition strength is
        # more aligned with the downstream goal than CFG-style training.
        #
        # Previous CFG training path is intentionally disabled here:
        # if p_uncond > 0:
        #     uncond_mask = torch.rand(indices.shape[0], device=device) < p_uncond
        #     if uncond_mask.any():
        #         _empty_context = torch.tensor(dataset.empty_context, device=device, dtype=context_cro.dtype)
        #         _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=indices.shape[0])
        #         context_cro = context_cro.clone()
        #         context_fin = context_fin.clone()
        #         context_cro[uncond_mask] = _empty_context[uncond_mask]
        #         context_fin[uncond_mask] = _empty_context[uncond_mask]
        #     _metrics['monitor/uncond_ratio'] = uncond_mask.float().mean().detach()
        # else:
        #     _metrics['monitor/uncond_ratio'] = torch.tensor(0.0, device=device)
        _metrics['monitor/uncond_ratio'] = torch.tensor(0.0, device=device)

        # 2. 计算损失：传入 switch_ratio 和 sigmoid_slope (可以从 config 中读取)
        loss_noise, loss_round, T_loss, log_info = utils.LSimple(
            _z, nnet, _schedule,
            context=[context_cro, context_fin],
            switch_ratio=config.get('switch_ratio', 0.5),
            sigmoid_slope=config.get('sigmoid_slope', 10),
            **_batch
        )

        # 3. 将监控字典合并到 _metrics 中
        _metrics.update(log_info)

        # 4. 记录原本的训练指标
        _metrics['loss_noise'] = loss_noise.detach().mean()
        _metrics['loss_mse'] = loss_noise.detach().mean()
        _metrics['loss_x0'] = log_info['monitor/loss_x0']
        scaled_cluster_loss = cluster_metrics['scaled_proto_nce_loss'].detach().mean()
        _metrics['loss_c'] = scaled_cluster_loss
        _metrics['monitor/loss_c_raw'] = cluster_metrics['raw_proto_nce_loss'].detach().mean()
        _metrics['monitor/loss_c_scaled'] = scaled_cluster_loss
        _metrics['loss_rounding'] = loss_round.detach().mean()
        _metrics['T_loss'] = T_loss.detach().mean()

        # 5. 总损失计算与反向传播 (逻辑维持原样)
        if epoch > config['warm']:
            total_loss = loss_noise.mean() + neigh_loss.mean() + loss_round.mean() + T_loss.mean()
        else:
            total_loss = loss_noise.mean() + loss_round.mean() + T_loss.mean()
            
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        
        # 返回包含所有监控指标的字典
        return _z, dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)


    # def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
    #     _z_init = torch.randn(_n_samples, *config['z_shape'], device=device) # 从正态分布中采样噪声。 # B,3,128
    #     z_init2 = kwargs['z_init']
    #     mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1,1, config['z_shape'][1]) # B,3,128
    #     _z_init = torch.where(mask, _z_init, z_init2) # 从正态分布中采样噪声。 # B,3,128

    #     context_cro, context_fin = kwargs['context']  # ← 关键！传入 [cro, fin]
    #     T = 1000  # 总步数（与训练一致）
    #     # switch_ratio = 0.5
    #     switch_ratio = config.get('switch_ratio', 0.5)  # 加上这行
    #     sigmoid_slope = config.get('sigmoid_slope', 10) # 加上这行

    #     # 先加噪声，重新定义schedule
    #     noise_schedule = NoiseScheduleVP(schedule='discrete',
    #                                      betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式

    #     # def model_fn(x, t_continuous):
    #     #     t = t_continuous * _schedule.N
    #     #     return cfg_nnet(x, t ,kwargs['context'])
    #     def model_fn(x, t_continuous):
    #         """
    #         x: 当前加噪状态 [B, 3, 128]
    #         t_continuous: DPM-Solver 中的连续时间 t ∈ [0,1]
    #         """
    #         t = t_continuous * T  # 转换为离散步数 [0, 1000]
    #         t_norm = t / T        # 归一化 [0,1]

    #         # --- 关键修改点 ---
    #         # 1. 计算注入强度：t_norm 越大（高噪声），sigmoid 越大，我们要让注入越小
    #         # 因此使用 1.0 - sigmoid
    #         # 当 t_norm < switch_ratio 时，weight_injection 趋近 1 (注入 Fine)
    #         # 当 t_norm > switch_ratio 时，weight_injection 趋近 0 (保留 Coarse)
    #         weight_injection = 1.0 - torch.sigmoid(sigmoid_slope * (t_norm - switch_ratio))
            
    #         # 2. 动态融合 context (采用与 LSimple 完全一致的残差形式)
    #         if context_cro.dim() == 3:  # [B, C, D]
    #             # 公式：Base + Weight * (Target - Base)
    #             context = context_cro + weight_injection[:, None, None] * (context_fin - context_cro)
    #         else:  # scalar weight
    #             context = context_cro + weight_injection * (context_fin - context_cro)
    #         # ------------------

    #         # 转换为离散时间步
    #         t_discrete = (t_continuous * T).long()
    #         t_discrete = torch.clamp(t_discrete, 0, T-1)

    #         return cfg_nnet(x, t_discrete, context)
        
    #     dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    #     _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)  # embedding， 感觉这里需要rounding
    #     return decode(_z, nnet_ema.get_logits)
    def dpm_solver_sample(_n_samples, _sample_steps, condition_mode='hybrid', **kwargs):
        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device) # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1,1, config['z_shape'][1]) 
        _z_init = torch.where(mask, _z_init, z_init2) 

        context_cro, context_fin = kwargs['context']  # [cro, fin]

        # Schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float()) 

        def model_fn(x, t_continuous):
            """
            x: 当前加噪状态 [B, 3, 128]
            t_continuous: DPM-Solver 中的连续时间 t ∈ [0,1]
            """
            t = t_continuous * 1000  # 转换为离散步数 [0, 1000]
            context = select_sampling_context(
                context_cro,
                context_fin,
                t,
                condition_mode=condition_mode,
            )

            # 转换为离散时间步
            t_discrete = (t_continuous * 1000).long()
            t_discrete = torch.clamp(t_discrete, 0, 999)

            return cfg_nnet(x, t_discrete, context)
        
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.) 
        return decode(_z, nnet_ema.get_logits)


    def eval_step(n_samples, sample_steps):
        """这里需要做case study"""
        # 逐步采样。这里用上了DPM加速
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config["sample"]["mini_batch_size"]}')
        eval_modes = ['hybrid']
        if config.get('eval_coarse_only', True):
            eval_modes.append('coarse_only')

        eval_results = {}
        base_path = config['sample']['path'] + data_n + '/' + task
        for condition_mode in eval_modes:
            def sample_fn(_n_samples):
                indices, batch = next(test_generator)
                assert indices.size(0) == _n_samples
                indices, sample_input = convert_input((indices, batch))
                _z, context_cro, context_fin, _, _, labels, hard_labels = nnet_ema.encode_everything_test(indices, **batch)
                return labels, hard_labels, dpm_solver_sample(
                    _n_samples,
                    sample_steps,
                    condition_mode=condition_mode,
                    context=[context_cro, context_fin],
                    z_init=_z,
                    mask=sample_input['mask'],
                )

            with tempfile.TemporaryDirectory() as temp_path:  # 创建临时目录，进行文件管理
                path = base_path if condition_mode == 'hybrid' else os.path.join(base_path, condition_mode)
                work_dir = config['workdir'] + data_n + '/' + task
                os.makedirs(path, exist_ok=True)
                os.makedirs(work_dir, exist_ok=True)
                utils.sample2dir(path, n_samples, config['sample']['mini_batch_size'], sample_fn, tokenizers,
                                 dataset.unpreprocess)

                _quality, jaccard_quality = utils.calculate_quality(path)
                metric_suffix = '' if condition_mode == 'hybrid' else f'_{condition_mode}'
                avg_jaccard = float(sum(jaccard_quality) / len(jaccard_quality))
                logging.info(f'step={train_state.step} quality{metric_suffix}-{n_samples}={_quality}')
                logging.info(f'step={train_state.step} jaccard_quality{metric_suffix}-{n_samples}={jaccard_quality}')
                with open(os.path.join(config['workdir'] + data_n + '/' + task, 'eval.log'), 'a') as f:
                    print(
                        f'step={train_state.step} quality{metric_suffix}-{n_samples}={_quality} '
                        f'jaccard_quality{metric_suffix}-{n_samples}={jaccard_quality}',
                        file=f
                    )
                wandb.log({
                    f'quality_mse{metric_suffix}-{n_samples}': _quality,
                    f'avg_jaccard{metric_suffix}-{n_samples}': avg_jaccard,
                    f'jaccard_conditions{metric_suffix}-{n_samples}': float(jaccard_quality[0]),
                    f'jaccard_procedures{metric_suffix}-{n_samples}': float(jaccard_quality[1]),
                    f'jaccard_drugs{metric_suffix}-{n_samples}': float(jaccard_quality[2]),
                }, step=train_state.step)
                eval_results[condition_mode] = (_quality, jaccard_quality)

        hybrid_quality, hybrid_jaccard = eval_results['hybrid']
        return hybrid_quality, hybrid_jaccard, eval_results

    # logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config["mixed_precision"]}')

    logging.info(f'Start fitting, step={train_state.step}')

    def convert_input(batch_data, mask_static=True):
        # padding到相同的长度便于转为tensor
        indices, batch = batch_data
        # print("batch_data 0", batch['mask'])
        feature_keys = ['conditions', 'procedures', 'drugs']
        # batch_new = {}
        for feature_key in feature_keys:
            # tokenizers
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key]) # B,M,  m——feature没有用。
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist']) # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp']) # B,M
            # tensor
            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
            batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)
            # mask
            batch[feature_key + '_mask'] = batch[feature_key] != 0 # 为True的就是mask掉的, # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0 # 这个是为了padding, # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0 # 这个是为了padding # B,M

            # others
            # tianzhi add 第一个是存储的原始的，token的，没有padding的；第二个存储的是原始的，没有token之前的原始ICD编码
            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(origin, padding=False,  truncation=False) # 不需要减去2,只需要能看清楚是什么就行，因为decode的时候会加上2
            batch[feature_key + '_comps_origin'] = origin # origin ID

            # note special
            batch['has_note'] = False
            # tianzhi add 修改这里的设置
            if config_health['DATASET'] == 'MIV-Note': # LOS MIV修改下。
                batch['note'] = torch.tensor(batch['note'], device=device) # cur, B,D; 这里不需要对他进行重建，因为他可能变为pad字符串
                batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)# hist, B,T,D
                batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) !=0 # B,T

                # 作为context,需要合并
                batch['note' + '_hist'] = torch.cat([batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1)  # B,T+1,D
                batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B,
                batch['note' + '_hist_mask'] = torch.cat(
                    [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)], dim=1)  # B,T+1

                batch['has_note'] = True

        # mask设置为固定
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        # mask不固定, evaldffs固定就好, diverse
        else:
            m = 3 # 4 if batch['has_note'] else 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0],m=m).to(device=device)         # tianzhi add 随机生成掩码矩阵，为一个批次的数据

        aligned_data = [batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for feature_key in feature_keys] # B,V,1
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B, V, 3，记录history mask状态

        return indices, batch


    step_quality = []
    jaccard_quality = []
    eval_records = []
    best_eval_mse = float('inf')
    bad_eval_count = 0
    early_stop_patience = int(config.get('early_stop_patience', 4))
    early_stop_min_delta = float(config.get('early_stop_min_delta', 0.2))
    should_stop = False
    # while train_state.step < config['train']['n_steps']:
    max_n_steps = (len(train_dataset) // config['train']['batch_size']) + 1 if len(train_dataset) % config['train']['batch_size'] !=0 else len(train_dataset) // config['train']['batch_size'] # 不对，如果正好整除的话，就不需要+1
    for epoch in range(config['train']['n_epochs']):
        if epoch == 0:
            visit_centroids, _ = nnet.e_step()  # 初始化
            nnet.update_centroid_emb(visit_centroids)
            sync_cluster_state(nnet, nnet_ema)
        # print("AAAAAAAAA", nnet.total_visit_emb.shape, max_n_steps, len(train_dataset))
        total_visit_embs = []
        total_indices = []
        for i in range(max_n_steps):
            nnet.train()
            # batch = tree_map(lambda x: convert_input(x), next(data_generator))  # tree_map更快
            # tianzhi add 这里是提取一个批次的数据，返回对应的索引和内容。
            indices, batch = convert_input(next(data_generator), mask_static=False)
            
            _z, metrics = train_step(indices, batch, epoch)
            total_visit_embs.append(_z.view(_z.shape[0],-1)) # B, 3D
            total_indices.append(indices) # B,1

            nnet.eval()
            # 记录metrics, 输出train_state.step, 以及config.train.log_interval
            # if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            # if train_state.step % config['train']['log_interval'] == 0:
            #     logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            #     logging.info(config['workdir'])
            #     wandb.log(metrics, step=train_state.step) # 不同step
            if train_state.step % 1000 == 0: 
                wandb.log(metrics, step=train_state.step)

            # 中途输出
            # if train_state.step % config['train']['eval_interval'] == 0:  # 任何记录日志，保存模型都要使用
            #     torch.cuda.empty_cache()
                logging.info('Save a grid of {} samples for training...'.format(config['train_val_num']))
                sample_input = dataset.get_contexts(num=config['train_val_num'])
                indices = torch.arange(config['train_val_num']).to(device)
                indices, sample_input = convert_input((indices, sample_input))
                # _z, contexts, _, labels, hard_labels = nnet.encode_everything_test(indices, **sample_input)
                _z, context_cro, context_fin, _, _, labels, hard_labels = nnet_ema.encode_everything_test(indices, **sample_input)
                
                # contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5] # 选取前10个context
                # samples, hard_samples = dpm_solver_sample(_n_samples=config['train_val_num'], _sample_steps=50, context=contexts, z_init=_z, mask=sample_input['mask'])
                samples, hard_samples = dpm_solver_sample(_n_samples=config['train_val_num'], _sample_steps=50, context=[context_cro,context_fin], z_init=_z, mask=sample_input['mask'])
                # print("generate", hard_samples['conditions'][0])
                # print("origin", tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])[0]) # 没啥问题，encode, decode用同一套
                # print("generate", hard_samples['procedures'][0])
                # print("origin", tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])[0])  # 没啥问题，encode, decode用同一套
                # print("generate", hard_samples['drugs'][0])
                # print("origin", tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])[0])  # 没啥问题，encode, decode用同一套
                # samples = make_grid(dataset.unpreprocess(samples), 5)
                # save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                # wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
                jaccard_condition = utils.calculate_average_jaccard(hard_samples['conditions'], tokenizers['conditions'].batch_decode_2d(hard_labels['conditions']))
                jaccard_procedure = utils.calculate_average_jaccard(hard_samples['procedures'], tokenizers['procedures'].batch_decode_2d(hard_labels['procedures']))
                jaccard_drug = utils.calculate_average_jaccard(hard_samples['drugs'], tokenizers['drugs'].batch_decode_2d(hard_labels['drugs']))
                mse_loss = F.mse_loss(samples, labels.squeeze())
                logging.info({'mask num': sample_input['mask'].sum(dim=0), 'val samples mse': mse_loss, 'jaccard': (jaccard_condition,jaccard_procedure,jaccard_drug)})
                # wandb.log({'samples': mse_loss }, step=train_state.step)
                # torch.cuda.empty_cache()
            # accelerator.wait_for_everyone()

            # save ckpt， 这一块代码检查当前进程是否为主进程（accelerator.is_main_process），如果是则保存模型
        # if train_state.step % config['train']['save_interval'] == 0 or train_state.step == config['train']['n_steps']:
        # torch.cuda.empty_cache()
        total_visit_embs = torch.cat(total_visit_embs, dim=0)
        total_indices = torch.cat(total_indices, dim=0)
        total_visit_embs = total_visit_embs[total_indices.argsort()] # 重新排序， 这里需要check下对不对。
        print("CCCCCCCC", total_visit_embs.shape)
        nnet.update_train_visit_emb(total_visit_embs)
        visit_centroids, _ = nnet.e_step()
        nnet.update_centroid_emb(visit_centroids)
        sync_cluster_state(nnet, nnet_ema)

        logging.info(f'Save and eval checkpoint {epoch}...')
        train_state.save(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch}.ckpt'))
        if epoch == 0 or (epoch >= config['train']['eval_epoch'] and epoch % config['train']['eval_epoch'] == 0):
            quality, jaccard_tuple, _ = eval_step(n_samples=len(test_dataset), sample_steps=50)
            eval_records.append({
                'epoch': epoch,
                'mse': float(quality),
                'jaccard': tuple(float(score) for score in jaccard_tuple),
                'avg_jaccard': float(sum(jaccard_tuple) / len(jaccard_tuple)),
            })
            if quality < best_eval_mse - early_stop_min_delta:
                best_eval_mse = float(quality)
                bad_eval_count = 0
            else:
                bad_eval_count += 1
                logging.info(
                    f'early_stop_counter={bad_eval_count}/{early_stop_patience}, '
                    f'best_eval_mse={best_eval_mse:.6f}, '
                    f'current_eval_mse={float(quality):.6f}'
                )
            if bad_eval_count >= early_stop_patience:
                logging.info(
                    f'Early stop triggered at epoch={epoch}, '
                    f'best_eval_mse={best_eval_mse:.6f}, '
                    f'patience={early_stop_patience}, '
                    f'min_delta={early_stop_min_delta}'
                )
                should_stop = True
        else:
            quality = None
            jaccard_tuple = None
        step_quality.append((epoch, quality))
        jaccard_quality.append((epoch, jaccard_tuple))
        if should_stop:
            break



    logging.info(f'Finish fitting, step={train_state.step}')

    if not eval_records:
        raise RuntimeError('No diffusion checkpoints were evaluated, cannot select a best model.')

    mse_quantile = float(config.get('model_select_mse_quantile', 0.3))
    mse_guard_ratio = float(config.get('model_select_mse_guard_ratio', 1.1))
    mse_values = np.array([record['mse'] for record in eval_records], dtype=np.float32)
    mse_cutoff_quantile = float(np.quantile(mse_values, mse_quantile))

    best_mse_record = min(eval_records, key=lambda record: record['mse'])
    best_jaccard_record = max(eval_records, key=lambda record: record['avg_jaccard'])
    mse_cutoff_guard = float(best_mse_record['mse'] * mse_guard_ratio)
    mse_cutoff = min(mse_cutoff_quantile, mse_cutoff_guard)

    hybrid_candidates = [record for record in eval_records if record['mse'] <= mse_cutoff]
    if hybrid_candidates:
        chosen_record = max(hybrid_candidates, key=lambda record: (record['avg_jaccard'], -record['mse']))
    else:
        chosen_record = best_mse_record

    epoch_best = chosen_record['epoch']
    logging.info(f'step_quality: {step_quality}')
    logging.info(f'jaccard_quality_history: {jaccard_quality}')
    logging.info(f'best_by_mse: {best_mse_record}')
    logging.info(f'best_by_jaccard: {best_jaccard_record}')
    logging.info(
        f'model_select_mse_quantile={mse_quantile}, '
        f'mse_guard_ratio={mse_guard_ratio}, '
        f'mse_cutoff_quantile={mse_cutoff_quantile}, '
        f'mse_cutoff_guard={mse_cutoff_guard}, '
        f'mse_cutoff={mse_cutoff}'
    )
    logging.info(f'hybrid_candidates: {hybrid_candidates}')
    logging.info(f'chosen_hybrid_model: {chosen_record}')

    train_state.load(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt'))
    if 'metrics' in locals():
        del metrics
    eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])
    nnet_path = os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt/nnet_ema.pth')
    return nnet_path, train_state.nnet_ema, train_state.nnet_ema.total_visit_emb.shape[0]
