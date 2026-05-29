# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : models.py
# Time       ：4/11/2024 3:24 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：
"""
import torch
import numpy as np
import torch.nn as nn

import itertools
import time
import pandas as pd
import torch
import dgl
import os
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional, Union
# from pyhealth.models.utils import get_last_visit
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.functional import multilabel_margin_loss
from pyhealth.metrics import ddi_rate_score
from pyhealth.models.utils import batch_to_multihot
from pyhealth.models import BaseModel
from pyhealth.medcode import ATC
from pyhealth.datasets import SampleEHRDataset
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from utils import get_last_visit
from utils import pad_list
import ipdb

class PredJoint(nn.Module):
    def __init__(self, feature_num, embedding_dim, nhead=4, dropout=0.1, voc_size=1000, task='REC'):
        super(PredJoint, self).__init__()
        rnns = torch.nn.TransformerEncoderLayer(
            d_model=feature_num * embedding_dim, nhead=nhead, batch_first=True, dropout=dropout)  # all others
        self.rnns = torch.nn.TransformerEncoder(rnns, num_layers=1)

        gru = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.gru = nn.TransformerEncoder(gru, num_layers=1)
        self.fina_proj = nn.Sequential(#nn.Dropout(dropout),# final的dropout只为eicu PHE，因为数据量太少了，容易过拟合。增加F1-score;降低Jaccard
                                       nn.Linear((feature_num+1) * embedding_dim,
                                                 voc_size))
        if task in ['PHE', 'DIAG', 'REC', 'MOR', 'REA','DRUG']:
            self.final_act = nn.Sigmoid()
        elif task in ['LOS']:
            self.final_act = nn.Softmax(dim=-1)

        self.clz_token = nn.Parameter(torch.randn(1, feature_num*embedding_dim))

    def add_clz_mask(self, mask):
        # 在掩码前添加 CLZ token
        batch_size, seq_length = mask.shape
        clz_mask = torch.ones(batch_size, 1).to(mask.device)  # CLZ token 的掩码为 1
        return torch.cat((clz_mask, mask), dim=1).bool()

    def add_clz_token(self, batch_data, clz_token):
        # 在每个句子前添加 CLZ token
        batch_size, seq_length, embed_size = batch_data.shape  # B,V,D
        # 创建 CLZ token的形状为 (B, 1, D)
        clz_token = clz_token.unsqueeze(0).expand(batch_size, -1, -1)
        # 拼接 CLZ token 和原始数据
        return torch.cat((clz_token, batch_data), dim=1)

    def forward(self, patient_emb, mask,patient_id, new_feature=None):
        # mask = self.add_clz_mask(mask)
        # patient_emb = self.add_clz_token(patient_emb, self.clz_token)
        patient_id = self.gru(patient_id, src_key_padding_mask=~mask)
        patient_id = get_last_visit(patient_id, mask)
        # patient_id = patient_id * mask.unsqueeze(dim=-1)
        # patient_id = torch.max(patient_id, dim=1).values#torch.sum(patient_id, dim=1)


        x = self.rnns(patient_emb, src_key_padding_mask=~mask)

        x = get_last_visit(x, mask)
        x = torch.cat([patient_id, x], dim=-1)

        if new_feature is not None:
            x = x + new_feature.view(new_feature.size(0),-1)
        logits = self.fina_proj(x)
        y_prob = self.final_act(logits)
        return logits, y_prob


class PredSingle(nn.Module):
    def __init__(self, embedding_dim,nhead=4, dropout=0.1, voc_size=1000, task='REC'):
        super(PredSingle, self).__init__()
        rnns = torch.nn.TransformerEncoderLayer(
            d_model= embedding_dim, nhead=nhead, batch_first=True, dropout=dropout)  # all others
        self.rnns = torch.nn.TransformerEncoder(rnns, num_layers=1)
        gru = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.gru = nn.TransformerEncoder(gru, num_layers=1)
        self.fina_proj = nn.Sequential(#nn.Dropout(dropout), # final的dropout只为eicu PHE，因为数据量太少了，容易过拟合。
                                       nn.Linear(((1+1))*embedding_dim,
                                                 voc_size))
        if task in ['PHE', 'DIAG', 'REC', 'MOR', 'REA','DRUG']:
            self.final_act = nn.Sigmoid()
        elif task in ['LOS']:
            self.final_act = nn.Softmax(dim=-1)
        self.clz_token = nn.Parameter(torch.randn(1,  embedding_dim))

    def add_clz_mask(self, mask):
        batch_size, seq_length = mask.shape
        clz_mask = torch.ones(batch_size, 1).to(mask.device)  # CLZ token 的掩码为 1
        return torch.cat((clz_mask, mask), dim=1).bool()

    def add_clz_token(self, batch_data, clz_token):
        # 在每个句子前添加 CLZ token
        batch_size, seq_length, embed_size = batch_data.shape  # B,V,D
        # 创建 CLZ token的形状为 (B, 1, D)
        clz_token = clz_token.unsqueeze(0).expand(batch_size, -1, -1)
        # 拼接 CLZ token 和原始数据
        return torch.cat((clz_token, batch_data), dim=1)

    def forward(self, x, mask, patient_id, new_feature=None):
        # mask = self.add_clz_mask(mask)
        # x = self.add_clz_token(x, self.clz_token)
        # patient_id = self.gru(patient_id, src_key_padding_mask=~mask)
        # patient_id = get_last_visit(patient_id, mask)

        x = self.rnns(x, src_key_padding_mask=~mask)
        x = get_last_visit(x, mask)
        x = torch.cat([patient_id, x], dim=-1)

        if new_feature is not None:
            x = x + new_feature
        logit = self.fina_proj(x)
        y_prob = self.final_act(logit)
        return logit, y_prob



########## for rebuttal
class PredSingle2(nn.Module):
    def __init__(self, embedding_dim,nhead=4, dropout=0.1, voc_size=1000, task='REC'):
        super(PredSingle2, self).__init__()
        rnns = torch.nn.TransformerEncoderLayer(
            d_model= embedding_dim, nhead=nhead, batch_first=True, dropout=dropout)  # all others
        # self.rnns = torch.nn.TransformerEncoder(rnns, num_layers=1)
        self.rnns = nn.GRU(embedding_dim, embedding_dim,num_layers=2, batch_first=True)
        gru = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.gru = nn.TransformerEncoder(gru, num_layers=1)
        self.fina_proj = nn.Sequential(#nn.Dropout(dropout), # final的dropout只为eicu PHE，因为数据量太少了，容易过拟合。
                                       nn.Linear(((1+1))*embedding_dim,
                                                 voc_size))
        if task in ['PHE', 'DIAG', 'REC', 'MOR', 'REA']:
            self.final_act = nn.Sigmoid()
        elif task in ['LOS']:
            self.final_act = nn.Softmax(dim=-1)
        self.clz_token = nn.Parameter(torch.randn(1,  embedding_dim))

    def add_clz_mask(self, mask):
        # 在掩码前添加 CLZ token
        batch_size, seq_length = mask.shape
        clz_mask = torch.ones(batch_size, 1).to(mask.device)  # CLZ token 的掩码为 1
        return torch.cat((clz_mask, mask), dim=1).bool()

    def add_clz_token(self, batch_data, clz_token):
        # 在每个句子前添加 CLZ token
        batch_size, seq_length, embed_size = batch_data.shape  # B,V,D
        # 创建 CLZ token的形状为 (B, 1, D)
        clz_token = clz_token.unsqueeze(0).expand(batch_size, -1, -1)
        # 拼接 CLZ token 和原始数据
        return torch.cat((clz_token, batch_data), dim=1)

    def forward(self, x, mask, patient_id, new_feature=None):
        # mask = self.add_clz_mask(mask)
        # x = self.add_clz_token(x, self.clz_token)
        # patient_id = self.gru(patient_id, src_key_padding_mask=~mask)
        # patient_id = get_last_visit(patient_id, mask)

        # x = self.rnns(x, src_key_padding_mask=~mask)
        x = self.rnns(x)[0]
        x = get_last_visit(x, mask)
        x = torch.cat([patient_id, x], dim=-1)

        if new_feature is not None:
            x = x + new_feature
        logit = self.fina_proj(x)
        y_prob = self.final_act(logit)
        return logit, y_prob



class PredSingle3(nn.Module):
    def __init__(self, embedding_dim,nhead=4, dropout=0.1, voc_size=1000, task='REC'):
        super(PredSingle3, self).__init__()
        rnns = torch.nn.TransformerEncoderLayer(
            d_model= embedding_dim, nhead=nhead, batch_first=True, dropout=dropout)  # all others
        # self.rnns = torch.nn.TransformerEncoder(rnns, num_layers=1)
        # self.rnns = nn.GRU(embedding_dim, embedding_dim,num_layers=2, batch_first=True)
        self.rnns = nn.MultiheadAttention(embedding_dim, num_heads=1, batch_first=True)
        gru = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.gru = nn.TransformerEncoder(gru, num_layers=1)
        self.fina_proj = nn.Sequential(#nn.Dropout(dropout), # final的dropout只为eicu PHE，因为数据量太少了，容易过拟合。
                                       nn.Linear(((1+1))*embedding_dim,
                                                 voc_size))
        if task in ['PHE', 'DIAG', 'REC', 'MOR', 'REA']:
            self.final_act = nn.Sigmoid()
        elif task in ['LOS']:
            self.final_act = nn.Softmax(dim=-1)
        self.clz_token = nn.Parameter(torch.randn(1,  embedding_dim))

    def add_clz_mask(self, mask):
        # 在掩码前添加 CLZ token
        batch_size, seq_length = mask.shape
        clz_mask = torch.ones(batch_size, 1).to(mask.device)  # CLZ token 的掩码为 1
        return torch.cat((clz_mask, mask), dim=1).bool()

    def add_clz_token(self, batch_data, clz_token):
        # 在每个句子前添加 CLZ token
        batch_size, seq_length, embed_size = batch_data.shape  # B,V,D
        # 创建 CLZ token的形状为 (B, 1, D)
        clz_token = clz_token.unsqueeze(0).expand(batch_size, -1, -1)
        # 拼接 CLZ token 和原始数据
        return torch.cat((clz_token, batch_data), dim=1)

    def forward(self, x, mask, patient_id, new_feature=None):
        # mask = self.add_clz_mask(mask)
        # x = self.add_clz_token(x, self.clz_token)
        # patient_id = self.gru(patient_id, src_key_padding_mask=~mask)
        # patient_id = get_last_visit(patient_id, mask)

        # x = self.rnns(x, src_key_padding_mask=~mask)
        x = self.rnns(x,x,x,key_padding_mask=~mask)[0]
        x = get_last_visit(x, mask)
        x = torch.cat([patient_id, x], dim=-1)

        if new_feature is not None:
            x = x + new_feature
        logit = self.fina_proj(x)
        y_prob = self.final_act(logit)
        return logit, y_prob


########## for rebuttal



class RecLayer(nn.Module):
    # multi emb inverse
    def __init__(self, embedding_dim, hidden_dim, voc_size, feature_key, dropout=0.1,nhead=4, config=None):
        super(RecLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.ddi_weight = 0.
        self.multiloss_weight = config['MULTI']
        self.aux = config['AUX']

        self.feature_key = feature_key
        self.feature_num = len(feature_key)

        self.id_proj = nn.Sequential(
            # nn.Dropout(0.7), # 提升jaccard,降低AUC
            nn.Linear(self.feature_num * embedding_dim, embedding_dim), # self.feature_num * for los miv
            # nn.Linear(embedding_dim//2, embedding_dim)
            )
        gru = nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=4, dropout=0.3, batch_first=True)
        self.gru = nn.TransformerEncoder(gru, num_layers=1)

        self.config=config
        self.pred_layer_subs = nn.ModuleDict({x: PredSingle(embedding_dim,nhead=4, dropout=0.3,voc_size=voc_size, task=self.config['TASK'])
                                             for x in feature_key}) # procedure: attention, procedure gru; gru: procedure attention


        #### for rebuttal
        # self.pred_layer_subs['procedures'] = PredSingle2(embedding_dim,nhead=4, dropout=0.3,voc_size=voc_size, task=self.config['TASK'])
        #### for rebuttal

        # self.pred_layer_2 = PredSingle(embedding_dim, nhead=nhead, dropout=dropout,voc_size=voc_size)
        # self.pred_layer_3 = PredSingle(embedding_dim, nhead=nhead, dropout=dropout,voc_size=voc_size)
        self.pred_joint = PredJoint(self.feature_num, embedding_dim,nhead=4, dropout=0.3,voc_size=voc_size, task=self.config['TASK'])
        # self.beta = np.ones((self.feature_num,))
        self.beta = np.ones((self.feature_num,))
        self.eta = config['AUX'] # decay; 这个用于学习n weight

        # 必须这样，不然有问题，因为roc_AUC指标计算报错
        task = self.config['TASK']
        if task in ['PHE', 'DIAG', 'REC', 'MOR', 'REA']:
            self.final_act = nn.Sigmoid()
        elif task in ['LOS']:
            self.final_act = nn.Softmax(dim=-1)



    def forward(
            self,
            patient_id: torch.Tensor,
            patient_emb: torch.Tensor,
            labels: torch.Tensor,
            ddi_adj: torch.Tensor,
            mask: torch.Tensor,
            labels_indexes: Optional[torch.Tensor] = None,
            soft_feature: torch.Tensor = None,
        ):
        # print("AAAAAAAAAA",patient_emb.shape)
        patient_id = self.id_proj(patient_id)       # tianzhi add [32,4,384] -> [32,4,128]
        patient_ids = self.gru(patient_id, src_key_padding_mask=~mask)# patient_id * mask.unsqueeze(dim=-1)  #
        # patient_ids = torch.max(patient_ids, dim=1).values# torch.sum(patient_ids, dim=1)

        patient_ids = get_last_visit(patient_ids, mask)

        logit_lis, y_prob_lis, loss_lis, loss_lis_copy = [], [], [], []
        for fea in self.feature_key:
            emb = patient_emb[fea]
            logit, y_prob = self.pred_layer_subs[fea](emb, mask, patient_ids)
            loss = self.calculate_loss(logit, y_prob, labels, labels_indexes)
            logit_lis.append(logit)
            y_prob_lis.append(y_prob)
            loss_lis.append(loss)
            loss_lis_copy.append(loss.item())
        joint_emb = torch.cat([patient_emb[feature] for feature in self.feature_key], dim=-1)
        logit_joint, y_prob_joint = self.pred_joint(joint_emb, mask, patient_id) # soft_feature
        loss_joint = self.calculate_loss(logit_joint,  y_prob_joint, labels, labels_indexes)

        loss_for_adv = np.array(loss_lis_copy)
        rel_adv = (loss_for_adv - loss_joint.item()) / loss_joint.item()
        self.beta = self.beta - self.eta * rel_adv
        for i in range(self.feature_num):
            self.beta[i] = max(0.1, self.beta[i])
        self.beta = self.beta / (sum(self.beta ** 2) ** (0.5))



        # value = np.array(loss_for_adv)
        # l1_norm = np.linalg.norm(value, ord=1)  # 计算L1范数
        # normalized_value_l1 = value / l1_norm  # 进行L1范数归一化
        # print("AAAAAA", normalized_value_l1)
        # joint_prob = y_prob1 * self.beta[0] + y_prob2 * self.beta[1] + y_prob3 * self.beta[2]
        # joint_loss = self.beta[0] * loss1 + self.beta[1] * loss2 + self.beta[2] * loss3
        ##### rebutall
        # self.beta = [2,1,1]#[10,1,1] (10,5,15) 从medication开始
        ###### rebutall

        joint_prob = sum(y_prob * beta for y_prob, beta in zip(y_prob_lis, self.beta))

        if self.config['TASK'] in ['LOS']:
            joint_prob = self.final_act(joint_prob) # 需要保证为1
        # joint_prob = self.final_act(joint_prob)

        # joint_loss = sum(beta * loss for beta, loss in zip(self.beta, loss_lis)) # 显示的用eta学习

        joint_loss = sum(beta * loss for beta, loss in zip(self.beta, loss_lis)) # 显示的用eta学习

        # 最后返回的是final pred
        # print("AAAAAAAA", joint_loss)
        return joint_loss, joint_prob

    def calculate_loss(
            self,
            logits: torch.Tensor,
            y_prob: torch.Tensor,
            labels: torch.Tensor,
            label_index: Optional[torch.Tensor] = None,
    ):
        if self.config['TASK'] in ['PHE', 'DIAG', 'REC', 'MOR', 'REA','DRUG']:
            loss_cls = binary_cross_entropy_with_logits(logits, labels)
        elif self.config['TASK'] in ['LOS']:
            loss_cls = F.cross_entropy(logits, labels)

            # #### for rebuttal
            # loss_cls = F.mse_loss(logits, labels) # if use it donnot 处理lable
            # #### for rebuttal


        if self.multiloss_weight > 0 and label_index is not None:
            loss_multi = multilabel_margin_loss(y_prob, label_index)
            loss_cls = self.multiloss_weight * loss_multi + (1 - self.multiloss_weight) * loss_cls

        return loss_cls


class Diffrm(BaseModel):
    # pyhealth通用接口
    def __init__(
            self,
            dataset: SampleEHRDataset,
            feature_keys=["conditions", "procedures", "drugs", "incomplete"],
            label_key="labels",
            mode="multilabel",

            # hyper related
            dropout: float = 0.3,
            num_rnn_layers: int = 2,
            embedding_dim: int = 64,
            hidden_dim: int = 64,
            **kwargs,
    ):
        super(Diffrm, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        # define
        self.num_rnn_layers = num_rnn_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.config = kwargs['config']

        self.dropout_id = torch.nn.Dropout(self.dropout)
        self.dropout_id2 = torch.nn.Dropout(0.7) # 0.7 forall except eicu phe=0.3
        self.diff_proj = nn.Sequential(nn.Linear(2*self.embedding_dim, self.embedding_dim))  # tianzhi add 把concatenate后的（原始emb+补全emb）变为原来的维度


        self.feat_tokenizers = self.get_feature_tokenizers()  # tokenizer
        self.label_tokenizer = self.get_label_tokenizer()  # 注意这里的drug可没有spec_token; 这里label索引需要加2对于正则化
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        # save ddi adj
        self.ddi_adj = torch.nn.Parameter(self.generate_ddi_adj(), requires_grad=False)
        ddi_adj = self.generate_ddi_adj()  # 用于存储
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())  # 计算ddi直接从这里读取

        # module
        self.feature_keys_subs = ['conditions', 'procedures', 'drugs'] if self.config['DATASET'] != 'MIV-Note' else ['conditions', 'procedures', 'drugs']
        self.rec_layer = RecLayer(self.embedding_dim, self.hidden_dim, self.label_size,
                                  feature_key=self.feature_keys_subs, dropout=dropout, config=self.config)

        # new for note dataset
        self.note_linear = nn.Linear(768, self.embedding_dim)
        self.incomplete_emb = nn.Embedding(2,self.embedding_dim)
        self.positional_enc = nn.Embedding(self.config['MAXSEQ'],len(self.feature_keys_subs)* self.embedding_dim)

        # self.apply(self.init_weights)

        # init params
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)  # ehr emb



    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True) # dataframe，这里使用了gamenet的ddi,不要存储
        # ddi = pd.read_csv('/home/czhaobo/KnowHealth/data/REC/MIII/processed/ddi_pairs.csv', header=0, index_col=0).values.tolist()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi # each row
        ]

        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

    def encode_patient(self, feature_key: str, raw_values: List[List[List[str]]], new_feature=0) -> torch.Tensor:
        """Encode patient data."""
        
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(raw_values,
                                                                  max_length=[self.config['MAXSEQ'],
                                                                              self.config['MAXCODESEQ']])  # 这里会padding, B,V,M
        codes = torch.tensor(codes, dtype=torch.long, device=self.device)
        masks = codes != 0  # B,V,M
        embeddings = self.embeddings[feature_key](codes)  # B,V,M,D
        embeddings = self.dropout_id(embeddings) # 为啥要dropout一下呢
    
        visit_emb = self.get_visit_emb(embeddings)  # B,V,D


        # new hard sample
        if new_feature:
            if self.config['MODEL'] == 'ours':
                new_codes = self.feat_tokenizers[feature_key].batch_encode_3d(new_feature,
                                                                          max_length=[self.config['MAXSEQ'],
                                                                                      self.config[
                                                                                          'MAXCODESEQ']])  # 这里会padding, B,V,M
                new_codes = torch.tensor(new_codes, dtype=torch.long, device=self.device)
                # masks = new_codes != 0  # B,V,M
                new_embeddings = self.embeddings[feature_key](new_codes)  # B,V,M,D
                new_embeddings = self.dropout_id2(new_embeddings)
                new_visit_emb = self.get_visit_emb(new_embeddings)  # B,V,D
                new_visit_emb = F.normalize(new_visit_emb, p=2, dim=-1)  # 这里是为了保证emb的范数为1  # 这里又是为了啥呢

            elif self.config['MODEL']=='MedDiff':
                new_visit_emb = torch.tensor(new_feature, dtype=torch.float, device=self.device)

            visit_emb = torch.cat([visit_emb, new_visit_emb], dim=-1) # 不行就只能viisit_emb
            visit_emb = self.diff_proj(visit_emb)

            # visit_emb = visit_emb + new_visit_emb

        return codes, embeddings, masks, visit_emb  # B,V, D



    def get_visit_emb(self, emb, feature_key=None, masks=None):
        emb = torch.sum(emb, dim=2)
        # emb = self.dropout_id(torch.sum(emb, dim=2))
        return emb

    def decode_label(self, array_prob, tokenizer):
        array_prob[array_prob >= 0.4] = 1
        array_prob[array_prob < 0.4] = 0  # 优化同步
        indices = [np.where(row == 1)[0].tolist() for row in array_prob]
        tokens = tokenizer.batch_decode_2d(indices)
        return tokens

    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=0.1)  # He initialization for Conv2d
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to zero
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization for Linear layers
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to zero



    def forward(
            self,
            patient_id: List[List[str]],
            conditions: List[List[List[str]]],  # 需要和dataset保持一致[名字，因为**data]
            procedures: List[List[List[str]]],
            drugs_hist: List[List[List[str]]],
            labels: List[List[str]],  # label
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.
        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels]
                    representing the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels]
                    representing the ground truth of each drug.
        """
        labels_copy = labels  # for case
        # # patient id
        # prepare labels
        if self.mode == "multilabel":
            labels_index = self.label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )  # [[23,32],[1,2,3]]，注意比feature_tokenizer少两位

            labels = batch_to_multihot(labels_index, self.label_size)  # tensor, B, Label_size;  # convert to multihot
            index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
            for idx, cont in enumerate(labels_index):
                # remove redundant labels
                cont = list(set(cont))
                index_labels[idx, : len(cont)] = cont  # remove padding and unk

            index_labels = torch.from_numpy(index_labels)  # 类似的！【23，38，39】
            labels = labels.to(self.device)  # for bce loss
            index_labels = index_labels.to(self.device)  # for multi label loss
        elif self.mode in ["multiclass", 'binary']:
            index_labels = None
            labels = self.prepare_labels(labels, self.label_tokenizer)



        # patient id
        # kwargs['miss_feature'] = torch.tensor(kwargs['miss_feature']).to(self.device)
        cond_code, _, condi_mask, condition_vis_emb = self.encode_patient("conditions",
                                                                          conditions, kwargs['new_conditions'])  # [B,V,M] [B,V,M,D]; [B,V,M], [B,V,D]
        proc_code, _, proc_mask, procedure_vis_emb = self.encode_patient("procedures", procedures, kwargs['new_procedures'])
        drug_code, _, drug_mask, drug_history_vis_emb = self.encode_patient("drugs",
                                                                            drugs_hist, kwargs['new_drugs'])  # drug rec的时候不能放drug 1，1，1，1
        # tianzhi add 这里返回的四个东西分别是 原始编码，原始embedding，mask（用于表示那些是填充，哪些是有意义的），visit-level embedding （补全后合并的数据，都合并了）
        
        # new for note
        if self.config['DATASET'] == 'MIV-Note':
            note_emb = self.note_linear(pad_list(kwargs['note'], device=self.device)) # B,T,D
            note_mask = note_emb.sum(dim=-1) != 0 # B,T

            seq_emb = {'conditions': condition_vis_emb+ note_emb, 'procedures': procedure_vis_emb, 'drugs': drug_history_vis_emb}
            mask = (torch.sum(condi_mask, dim=-1) + torch.sum(proc_mask, dim=-1) + torch.sum(drug_mask, dim=-1)) != 0 # + note_mask torch.sum(condi_mask, dim=-1) != 0  # visit-level mask; 这个更安全，emb相加可能为0
        else:
            seq_emb = {'conditions': condition_vis_emb, 'procedures': procedure_vis_emb, 'drugs': drug_history_vis_emb}
            mask = (torch.sum(condi_mask, dim=-1) + torch.sum(proc_mask, dim=-1) + torch.sum(drug_mask, dim=-1)) != 0
        
        patient_emb = seq_emb

        # # 加入mask emb, 仅对eicu-los有用
        # incomplete = pad_list(kwargs['incomplete'], device = self.device)
        # incomplete_emb = self.incomplete_emb(incomplete) # B,3,D
        # incomplete_emb = {'conditions': incomplete_emb[:,:,0,:], 'procedures': incomplete_emb[:,:,1,:], 'drugs':  incomplete_emb[:,:,2,:]}
        # patient_emb = {key: value + incomplete_emb[key] for key, value in patient_emb.items()}


        patient_id = torch.cat(
            [seq_emb[feature] for feature in self.feature_keys_subs], dim=-1)
        # tianzhi add 这里是为了将所有的visit-level embedding合并成一个大的embedding，作为患者表示

        # time = torch.arange(patient_id.shape[1], device=self.device).unsqueeze(0).expand(patient_id.shape[0], -1) # 仅对eICU PHE有用
        # time_emb = self.positional_enc(time)
        # patient_id = patient_id + time_emb

        # calculate loss
        loss, y_prob = self.rec_layer(  # patient
            patient_id,
            patient_emb=patient_emb,
            labels=labels,
            ddi_adj=self.ddi_adj,
            mask=mask,
            labels_indexes=index_labels,
            soft_feature=kwargs['miss_feature']
        )

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
            "labels_copy": labels_copy,  # case时候打开
        }

###########################################################################


# class VisitFusion(nn.Module):
#     def __init__(self, diag_dim, proc_dim, fused_dim):
#         super(VisitFusion, self).__init__()
#         self.diag_proj = nn.Linear(diag_dim, fused_dim)
#         self.proc_proj = nn.Linear(proc_dim, fused_dim)
#         self.attention = nn.MultiheadAttention(fused_dim, num_heads=4)
#         self.fused_dim = fused_dim
#     def forward(self, diag_emb, proc_emb, diag_mask, proc_mask):
#         # Project embeddings
#         diag_emb = self.diag_proj(diag_emb)  # [B, V, M1, fused_dim]
#         proc_emb = self.proc_proj(proc_emb)  # [B, V, M2, fused_dim]
        
#         # Masked mean aggregation
#         def masked_mean(tensor, mask, dim):
#             masked_tensor = tensor * mask.unsqueeze(-1)
#             sum_tensor = torch.sum(masked_tensor, dim=dim)
#             count = torch.sum(mask.unsqueeze(-1), dim=dim) + 1e-8
#             return sum_tensor / count
        
#         diag_agg = masked_mean(diag_emb, diag_mask, dim=2)  # [B, V, fused_dim]
#         proc_agg = masked_mean(proc_emb, proc_mask, dim=2)  # [B, V, fused_dim]
        
#         # Create per-visit sequences 把患者每次访问对应的就诊和手术拼接起来
#         fused_seq = torch.stack([diag_agg, proc_agg], dim=2)  # [B, V, 2, fused_dim]
        
#         # Reshape for attention
#         B, V = diag_agg.shape[0], diag_agg.shape[1]
#         N = B * V
#         D = self.fused_dim
#         fused_seq_reshaped = fused_seq.view(N, 2, D)  # [N, 2, D]
        
#         # Transpose for MultiheadAttention
#         fused_seq_transposed = fused_seq_reshaped.transpose(0, 1)  # [2, N, D]
        
#         # Create key_padding_mask
#         diag_visit_mask = torch.any(diag_mask, dim=2)  # [B, V]
#         proc_visit_mask = torch.any(proc_mask, dim=2)  # [B, V]
#         mask_per_visit = torch.stack([~diag_visit_mask, ~proc_visit_mask], dim=2)  # [B, V, 2]
#         key_padding_mask = mask_per_visit.view(N, 2)  # [N, 2]
        
#         # Apply attention
#         attn_output, _ = self.attention(fused_seq_transposed, fused_seq_transposed, fused_seq_transposed, key_padding_mask=key_padding_mask)
        
#         # Reshape attention output back to per-visit embeddings
#         attn_output = attn_output.transpose(0, 1)  # [N, 2, D]
#         visit_emb_per_n = attn_output.mean(dim=1)  # [N, D]
#         visit_emb = visit_emb_per_n.view(B, V, D)  # [B, V, D]
        
#         visit_emb_mask = diag_visit_mask | proc_visit_mask  # [B, V]

#         return visit_emb, visit_emb_mask


# class PatientRepresentation(nn.Module):
#     def __init__(self, visit_dim, num_heads=4, dropout=0.1):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=visit_dim,
#             nhead=num_heads,
#             dropout=dropout,
#             activation="gelu",
#             layer_norm_eps=1e-8,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.norm = nn.LayerNorm(visit_dim)

#     def forward(self, visit_emb, mask=None):
#         # 自动生成掩码（如果未提供）
#         if mask is None:
#             mask = visit_emb.sum(dim=-1) != 0  # [B, V]
#         mask = torch.tensor(mask, dtype=torch.bool)  # 确保为布尔型

#         visit_emb = torch.nan_to_num(visit_emb, nan=0.0, posinf=1e8, neginf=-1e8)  # 替换异常值
#         # Transformer编码
#         patient_emb = self.transformer(visit_emb, src_key_padding_mask=~mask)

#         # 动态选择最后一个有效就诊
#         last_valid_idx = mask.sum(dim=1) - 1  # [B]
#         patient_emb = patient_emb.gather(
#             1, 
#             last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, visit_emb.size(-1))
#         ).squeeze(1)

#         return self.norm(patient_emb)


# class PatientAggregator(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    
#     def forward(self, x, mask):
#         B, V, D = x.shape
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
#         x = torch.cat((cls_tokens, x), dim=1)  # [B, V+1, D]
#         cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
#         full_mask = torch.cat((cls_mask, mask), dim=1)  # [B, V+1]
#         src_key_padding_mask = ~full_mask  # [B, V+1]
#         output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, V+1, D]
#         return output[:, 0, :]  # [B, D], cls output


# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv
# from torch_geometric.utils import dense_to_sparse
# import pickle
# import os
# from torch.nn.parameter import Parameter

# class DrugEmbeddingModule(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, ddi_pkl_path, device = None):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.device = device if device is not None else torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
#         # 加载 DDI 邻接矩阵
#         self.ddi_adj = self._load_ddi_adj(ddi_pkl_path).to(self.device)
        
#         # 基础药物嵌入
#         self.base_embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        
#         # DDI 图编码器
#         self.ddi_gcn = GCNConv(embedding_dim, embedding_dim).to(self.device)
        
#         # 融合层
#         self.fusion = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim),
#             nn.ReLU(),
#             nn.LayerNorm(embedding_dim)
#         ).to(self.device)
        
#         # 预计算嵌入
#         self.drug_embeddings = self._precompute_drug_embeddings()
    
#     def _load_ddi_adj(self, ddi_pkl_path):
#         """从本地 pkl 文件加载 DDI 邻接矩阵"""
#         if ddi_pkl_path and os.path.exists(ddi_pkl_path):
#             with open(ddi_pkl_path, 'rb') as f:
#                 ddi_adj = pickle.load(f)
#             if not isinstance(ddi_adj, torch.Tensor):
#                 ddi_adj = torch.FloatTensor(ddi_adj)
#             if ddi_adj.shape != (self.vocab_size, self.vocab_size):
#                 raise ValueError(f"DDI adjacency matrix shape {ddi_adj.shape} does not match vocab_size {self.vocab_size}")
#             return ddi_adj
#         else:
#             # 默认全零矩阵
#             return torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.float)
    
#     def _precompute_drug_embeddings(self):
#         """预计算所有药物的 DDI 嵌入"""
#         embeddings = []
#         base_embs = self.base_embedding(torch.arange(self.vocab_size).to(self.device))  # [vocab_size, embedding_dim]
        
#         # DDI 图嵌入
#         ddi_edge_index = dense_to_sparse(self.ddi_adj)[0].to(self.device)
#         ddi_embs = self.ddi_gcn(base_embs, ddi_edge_index)  # [vocab_size, embedding_dim]
        
#         for drug_id in range(self.vocab_size):
#             fused_emb = self.fusion(ddi_embs[drug_id])
#             embeddings.append(fused_emb.to(self.device))
        
#         return nn.Parameter(torch.stack(embeddings).to(self.device), requires_grad=True)
    
#     def forward(self, drug_ids):
#         """根据药物ID获取融合嵌入"""
#         return self.drug_embeddings[drug_ids]

# class DrugRecommendationLayer(nn.Module):
#     def __init__(self, patient_dim, drug_dim, num_drugs, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.patient_proj = nn.Linear(patient_dim, drug_dim)
#         self.drug_embeddings = nn.Embedding(num_drugs, drug_dim)
        
#     def forward(self, patient_repr, drug_ids=None):
#         """
#         patient_repr: [B, D_p] 患者表示
#         drug_ids: [B, num_drugs] 可选，用于训练时的负采样
#         """
#         # 投影患者表示到药物空间
#         patient_proj = self.patient_proj(patient_repr)  # [B, D_d]
#         patient_proj = F.normalize(patient_proj, p=2, dim=-1)  # L2归一化
        
#         # 获取药物嵌入
#         drug_embs = self.drug_embeddings(torch.arange(self.drug_embeddings.num_embeddings).to(patient_repr.device))
#         drug_embs = F.normalize(drug_embs, p=2, dim=-1)  # [num_drugs, D_d]
        
#         # 计算相似度
#         similarity = torch.matmul(patient_proj, drug_embs.t())  # [B, num_drugs]
#         similarity = similarity / self.temperature
        
#         # 计算概率
#         y_prob = torch.sigmoid(similarity)  # 多标签分类使用sigmoid
        
#         # 如果提供了drug_ids，计算损失
#         loss = None
#         if drug_ids is not None:
#             # 创建目标：正样本为1，负样本为0
#             target = torch.zeros_like(similarity, dtype=torch.float32, device=similarity.device)
#             drug_ids = drug_ids.to(torch.long)
#             for i, drugs in enumerate(drug_ids):
#                 target[i, drugs] = 1
                
#             # 计算二元交叉熵损失
#             loss = F.binary_cross_entropy_with_logits(similarity, target)
        
#         return loss, y_prob






#     def forward(
#             self,
#             patient_id: torch.Tensor,
#             patient_emb: torch.Tensor,
#             labels: torch.Tensor,
#             ddi_adj: torch.Tensor,
#             mask: torch.Tensor,
#             labels_indexes: Optional[torch.Tensor] = None,
#             soft_feature: torch.Tensor = None,
#         ):
#         # print("AAAAAAAAAA",patient_emb.shape)
#         patient_id = self.id_proj(patient_id)       # tianzhi add [32,4,384] -> [32,4,128]
#         patient_ids = self.gru(patient_id, src_key_padding_mask=~mask)# patient_id * mask.unsqueeze(dim=-1)  #
#         # patient_ids = torch.max(patient_ids, dim=1).values# torch.sum(patient_ids, dim=1)

#         patient_ids = get_last_visit(patient_ids, mask)

#         logit_lis, y_prob_lis, loss_lis, loss_lis_copy = [], [], [], []
#         for fea in self.feature_key:
#             emb = patient_emb[fea]
#             logit, y_prob = self.pred_layer_subs[fea](emb, mask, patient_ids)
#             loss = self.calculate_loss(logit, y_prob, labels, labels_indexes)
#             logit_lis.append(logit)
#             y_prob_lis.append(y_prob)
#             loss_lis.append(loss)
#             loss_lis_copy.append(loss.item())
#         joint_emb = torch.cat([patient_emb[feature] for feature in self.feature_key], dim=-1)
#         logit_joint, y_prob_joint = self.pred_joint(joint_emb, mask, patient_id) # soft_feature
#         loss_joint = self.calculate_loss(logit_joint,  y_prob_joint, labels, labels_indexes)

#         loss_for_adv = np.array(loss_lis_copy)
#         rel_adv = (loss_for_adv - loss_joint.item()) / loss_joint.item()
#         self.beta = self.beta - self.eta * rel_adv
#         for i in range(self.feature_num):
#             self.beta[i] = max(0.1, self.beta[i])
#         self.beta = self.beta / (sum(self.beta ** 2) ** (0.5))


#         joint_prob = sum(y_prob * beta for y_prob, beta in zip(y_prob_lis, self.beta))

#         if self.config['TASK'] in ['LOS']:
#             joint_prob = self.final_act(joint_prob) # 需要保证为1
       

#         joint_loss = sum(beta * loss for beta, loss in zip(self.beta, loss_lis)) # 显示的用eta学习

#         return joint_loss, joint_prob

#     def calculate_loss(
#             self,
#             logits: torch.Tensor,
#             y_prob: torch.Tensor,
#             labels: torch.Tensor,
#             label_index: Optional[torch.Tensor] = None,
#     ):
#         if self.config['TASK'] in ['PHE', 'DIAG', 'REC', 'MOR', 'REA','DRUG']:
#             loss_cls = binary_cross_entropy_with_logits(logits, labels)
#         elif self.config['TASK'] in ['LOS']:
#             loss_cls = F.cross_entropy(logits, labels)


#         if self.multiloss_weight > 0 and label_index is not None:
#             loss_multi = multilabel_margin_loss(y_prob, label_index)
#             loss_cls = self.multiloss_weight * loss_multi + (1 - self.multiloss_weight) * loss_cls

#         return loss_cls

# class GCN(nn.Module):
#     def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu')):
#         super(GCN, self).__init__()
#         self.voc_size = voc_size
#         self.emb_dim = emb_dim
#         self.device = device

#         ehr_adj = ehr_adj.cpu().numpy()  # Convert to NumPy array on CPU
#         ddi_adj = ddi_adj.cpu().numpy()
#         # Normalize adjacency matrices
#         ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
#         ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

#         # Move adjacency matrices and identity matrix to the specified device
#         self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
#         self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
#         self.x = torch.eye(voc_size).to(device)

#         # Initialize GCN layers and move them to the specified device
#         self.gcn1 = GraphConvolution(voc_size, emb_dim).to(device)
#         self.dropout = nn.Dropout(p=0.3).to(device)
#         self.gcn2 = GraphConvolution(emb_dim, emb_dim).to(device)
#         self.gcn3 = GraphConvolution(emb_dim, emb_dim).to(device)
#         ################
#         # print(f"self.x device: {self.x.device}")
#         # print(f"self.ehr_adj device: {self.ehr_adj.device}")
#     def forward(self):
#         # Ensure input and model parameters are on the same device
#         ehr_node_embedding = F.relu(self.gcn1(self.x, self.ehr_adj))
#         ddi_node_embedding = F.relu(self.gcn1(self.x, self.ddi_adj))

#         ehr_node_embedding = self.dropout(ehr_node_embedding)
#         ddi_node_embedding = self.dropout(ddi_node_embedding)

#         ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
#         ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
#         return ehr_node_embedding, ddi_node_embedding

#     def normalize(self, mx):
#         """Row-normalize sparse matrix"""
#         rowsum = np.array(mx.sum(1))
#         r_inv = np.power(rowsum, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = np.diagflat(r_inv)
#         mx = r_mat_inv.dot(mx)
#         return mx

# class GraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, in_features, out_features, bias=True, device=torch.device('cpu')):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.device = device
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features)).to(device)
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features)).to(device)
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):
#         # print(f"input device: {input.device}, adj device: {adj.device}, weight device: {self.weight.device}")
#         support = torch.mm(input, self.weight)
#         output = torch.mm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'
    
# # class DrugRecommendationModel(nn.Module):
# #     def __init__(self, num_drugs, embedding_dim):
# #         super().__init__()
# #         self.drug_embeddings = nn.Parameter(torch.randn(num_drugs, embedding_dim))
# #         nn.init.xavier_uniform_(self.drug_embeddings)   # 稳定初始化

# #     def forward(self, final_repr):
# #         # 1. 加数值裁剪防止极端 logits
# #         scores = torch.mm(final_repr, self.drug_embeddings.T)
# #         scores = torch.clamp(scores, min=-10, max=10)     # 避免 ±inf
# #         return scores                                     # 只返回 logits

# #     def calculate_loss(self, scores, y_true):
# #         # 2. 保证 target 无 nan
# #         y_true = torch.nan_to_num(y_true, nan=0.0)
# #         loss = nn.functional.binary_cross_entropy_with_logits(
# #             scores, y_true, reduction='mean')
# #         return loss

# class DrugRecommendationModel(nn.Module):
#     def __init__(self, num_drugs, embedding_dim):
#         super(DrugRecommendationModel, self).__init__()
#         self.drug_embeddings = nn.Parameter(torch.randn(num_drugs, embedding_dim))
#         self.init_weights()

#     def init_weights(self):
#         nn.init.normal_(self.drug_embeddings, mean=0, std=1e-3)  # 小方差初始化
#         self.drug_embeddings.register_hook(
#             lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
#         )

#     def forward(self, final_repr):
#         scores = torch.mm(final_repr, self.drug_embeddings.T)
#         scores = torch.clamp(scores, min=-10, max=10)     # 避免 ±inf
#         return scores

#     def calculate_loss(self, scores, y_true):
#         loss_fn = nn.BCEWithLogitsLoss()
#         loss = loss_fn(scores, y_true)
#         return loss

# from pyhealth.tokenizer import Tokenizer

# class Diffrm(BaseModel):
#     # pyhealth通用接口
#     def __init__(
#             self,
#             dataset: SampleEHRDataset,
#             feature_keys=["conditions", "procedures", "drugs", "incomplete"],
#             label_key="labels",
#             mode="multilabel",

#             # hyper related       
#             dropout: float = 0.3,
#             num_rnn_layers: int = 2,
#             embedding_dim: int = 64,
#             hidden_dim: int = 64,
#             **kwargs,
#     ):
#         super(Diffrm, self).__init__(
#             dataset=dataset,
#             feature_keys=feature_keys,
#             label_key=label_key,
#             mode=mode,
#         )
#         # define
#         self.num_rnn_layers = num_rnn_layers
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.config = kwargs['config']

#         # tianzhi add 用于控制历史和当前的权重
#         self.alpha = nn.Parameter(torch.tensor(0.5))

#         self.dropout_id = torch.nn.Dropout(self.dropout)
#         self.dropout_id2 = torch.nn.Dropout(0.7) # 0.7 forall except eicu phe=0.3
#         self.diff_proj = nn.Sequential(nn.Linear(2*self.embedding_dim, self.embedding_dim))  # tianzhi add 这里是干什么的

#         # tianzhi add
#         self.feat_tokenizers = self.get_all_tokenizers(dataset, special_tokens=True)  # tokenizer
#         # self.feat_tokenizers = self.get_feature_tokenizers()  # tokenizer
#         self.label_tokenizer = self.get_label_tokenizer()  # 注意这里的drug可没有spec_token; 这里label索引需要加2对于正则化
#         self.label_size = self.label_tokenizer.get_vocabulary_size()

#         # save ddi adj
#         # self.ddi_adj = torch.nn.Parameter(self.generate_ddi_adj(), requires_grad=False)
#         # ddi_adj = self.generate_ddi_adj()  # 用于存储
#         # np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())  # 计算ddi直接从这里读取
#         self.vocab_size = len(self.feat_tokenizers['drugs'].vocabulary.idx2token)

#         # module
#         self.fusion = VisitFusion(self.embedding_dim, self.embedding_dim, self.embedding_dim)       # 128 128 128 
#         self.patient_representation = PatientRepresentation(self.embedding_dim, num_heads=4)  # 128 4
#         # self.DrugEmbeddingModule = DrugEmbeddingModule(len(self.feat_tokenizers['drugs'].vocabulary.idx2token), self.embedding_dim, ddi_pkl_path="/home/user4/tianzhi/pyHealth_drug/data/ddi_A_final.pkl")
#         # self.drug_rec_layer = DrugRecommendationLayer(
#         #     patient_dim=embedding_dim,  # 患者表示的维度
#         #     drug_dim=embedding_dim,     # 药物嵌入的维度
#         #     num_drugs=self.label_size   # 药物词汇表大小
#         # )
#         self.DrugRecommendationModel = DrugRecommendationModel(num_drugs=192, embedding_dim=self.embedding_dim)
#         self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         # ipdb.set_trace()
#         # self.ddi_adj = self._load_ddi_adj("/home/user4/tianzhi/pyHealth_drug/data/ddi_A_final.pkl").to(self._device)
#         # self.gcn = GCN(voc_size=len(self.feat_tokenizers['drugs'].vocabulary.idx2token)-2, emb_dim=self.embedding_dim, ehr_adj=self.ddi_adj, ddi_adj=self.ddi_adj, device=self._device)
#         self.feature_keys_subs = ['conditions', 'procedures', 'drugs'] if self.config['DATASET'] != 'MIV-Note' else ['conditions', 'procedures', 'drugs']


#         # new for note dataset
#         self.note_linear = nn.Linear(768, self.embedding_dim)
#         self.incomplete_emb = nn.Embedding(2,self.embedding_dim)
#         self.positional_enc = nn.Embedding(self.config['MAXSEQ'],len(self.feature_keys_subs)* self.embedding_dim)

#         label_to_drug = self.feat_tokenizers['drugs'].vocabulary.idx2token
#         self.vocab = self.feat_tokenizers['drugs'].vocabulary.token2idx
#         self.drug_voc_indices = torch.tensor(
#             [self.vocab[label_to_drug[i]] for i in range(self.vocab_size)],
#             dtype=torch.long
#         ).to(self._device)

#         # self.apply(self.init_weights)

#         # init params
#         self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)  # ehr emb
#         self.apply(self.init_weights)

#         # 嵌入层稳定初始化和梯度钩子
#         for emb in self.embeddings.values():
#             nn.init.normal_(emb.weight, mean=0, std=1e-3)
#             emb.weight.register_hook(
#                 lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
#             )




#     def generate_ddi_adj(self) -> torch.FloatTensor:
#         """Generates the DDI graph adjacency matrix."""
#         atc = ATC()
#         ddi = atc.get_ddi(gamenet_ddi=True) # dataframe，这里使用了gamenet的ddi,不要存储
#         # ddi = pd.read_csv('/home/czhaobo/KnowHealth/data/REC/MIII/processed/ddi_pairs.csv', header=0, index_col=0).values.tolist()
#         vocab_to_index = self.label_tokenizer.vocabulary
#         ddi_adj = np.zeros((self.label_size, self.label_size))
#         ddi_atc3 = [
#             [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi # each row
#         ]

#         for atc_i, atc_j in ddi_atc3:
#             if atc_i in vocab_to_index and atc_j in vocab_to_index:
#                 ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
#                 ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
#         ddi_adj = torch.FloatTensor(ddi_adj)
#         return ddi_adj

#     def encode_patient(self, feature_key: str, raw_values: List[List[List[str]]], new_feature=0) -> torch.Tensor:
#         """Encode patient data."""
        
#         codes = self.feat_tokenizers[feature_key].batch_encode_3d(raw_values,
#                                                                   max_length=[self.config['MAXSEQ'],
#                                                                               self.config['MAXCODESEQ']])  # 这里会padding, B,V,M
#         codes = torch.tensor(codes, dtype=torch.long, device=self.device)
#         masks = codes != 0  # B,V,M
#         embeddings = self.embeddings[feature_key](codes)  # B,V,M,D
#         embeddings = self.dropout_id(embeddings)

#         visit_emb = self.get_visit_emb(embeddings)  # B,V,D
#         # print("codes shape", codes.shape, "max idx", codes.max().item())
#         # print("embedding table size", self.embeddings[feature_key].weight.shape[0])
#         # print("codes nan?", torch.isnan(codes.float()).any())
#         # print("embedding weight nan?", torch.isnan(self.embeddings[feature_key].weight).any())

#         # new hard sample
#         if new_feature:
#             if self.config['MODEL'] == 'ours':
#                 new_codes = self.feat_tokenizers[feature_key].batch_encode_3d(new_feature,
#                                                                           max_length=[self.config['MAXSEQ'],
#                                                                                       self.config[
#                                                                                           'MAXCODESEQ']])  # 这里会padding, B,V,M
#                 new_codes = torch.tensor(new_codes, dtype=torch.long, device=self.device)
#                 # masks = new_codes != 0  # B,V,M
#                 new_embeddings = self.embeddings[feature_key](new_codes)  # B,V,M,D
#                 new_embeddings = self.dropout_id2(new_embeddings)
#                 new_visit_emb = self.get_visit_emb(new_embeddings)  # B,V,D
#                 new_visit_emb = F.normalize(new_visit_emb, p=2, dim=-1)  # 这里是为了保证emb的范数为1

#             elif self.config['MODEL']=='MedDiff':
#                 new_visit_emb = torch.tensor(new_feature, dtype=torch.float, device=self.device)

#             visit_emb = torch.cat([visit_emb, new_visit_emb], dim=-1) # 不行就只能viisit_emb
#             visit_emb = self.diff_proj(visit_emb)

#             # visit_emb = visit_emb + new_visit_emb

#         return codes, embeddings, masks, visit_emb  # B,V, D
#                 # codes 分词、填充、转为tensor  
#                 # embeddings 是code的稠密向量 
#                 # masks 用于表示填充后的codes每个位置是否有效 
#                 # visit_emb 融合了补全特征的嵌入向量

#     def get_all_tokenizers(self, dataset, special_tokens=False):
#         if not special_tokens:
#             special_tokens = ["<pad>", "<unk>"] # 把pad取消
#         feature_keys = ["conditions", "procedures", "drugs"]
#         feature_tokenizers = {}
#         for feature_key in feature_keys:
#             tokens1 = dataset.get_all_tokens(key=feature_key)
#             if feature_key == "conditions":
#                 tokens2 = dataset.get_all_tokens(key="next_conditions")
#             elif feature_key == "procedures":
#                 tokens2 = dataset.get_all_tokens(key="next_procedures")
#             elif feature_key == "drugs":
#                 tokens2 = dataset.get_all_tokens(key="labels")
#             tokens = tokens1 + tokens2
#             feature_tokenizers[feature_key] = Tokenizer(
#                 tokens=tokens,
#                 special_tokens=None,
#             )
#             print(feature_key, feature_tokenizers[feature_key].get_vocabulary_size())
#         return feature_tokenizers
#     def get_visit_emb(self, emb, feature_key=None, masks=None):
#         emb = torch.sum(emb, dim=2)
#         # emb = self.dropout_id(torch.sum(emb, dim=2))
#         return emb

#     def decode_label(self, array_prob, tokenizer):
#         array_prob[array_prob >= 0.4] = 1
#         array_prob[array_prob < 0.4] = 0  # 优化同步
#         indices = [np.where(row == 1)[0].tolist() for row in array_prob]
#         tokens = tokenizer.batch_decode_2d(indices)
#         return tokens

#     # def init_weights(self,m):
#     #     if isinstance(m, nn.Conv2d):
#     #         nn.init.kaiming_uniform_(m.weight, a=0.1)  # He initialization for Conv2d
#     #         if m.bias is not None:
#     #             nn.init.constant_(m.bias, 0)  # Initialize bias to zero
#     #     elif isinstance(m, nn.Linear):
#     #         nn.init.xavier_uniform_(m.weight)  # Xavier initialization for Linear layers
#     #         if m.bias is not None:
#     #             nn.init.constant_(m.bias, 0)  # Initialize bias to zero
#     def init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Embedding):
#             nn.init.normal_(m.weight, mean=0, std=1e-3)

#     def calculate_ddi_loss(self, y_prob, ddi_adj):
#         """
#         计算DDI损失，惩罚可能发生药物相互作用的组合
#         y_prob: [B, num_drugs] 药物预测概率
#         ddi_adj: [num_drugs, num_drugs] DDI邻接矩阵
#         """
#         # 获取批次中每个患者的预测药物组合
#         drug_comb = torch.sigmoid(y_prob) > 0.5  # 阈值化
        
#         # 计算每对药物之间的DDI得分
#         batch_ddi_loss = 0
#         for i in range(y_prob.size(0)):
#             # 获取当前患者的药物组合
#             drugs = torch.where(drug_comb[i])[0]
            
#             # 计算这些药物之间的DDI得分
#             if len(drugs) > 0:
#                 drug_subset = drugs
#                 sub_adj = ddi_adj[drug_subset][:, drug_subset]
#                 ddi_score = torch.sum(sub_adj) / (len(drugs) * (len(drugs) - 1)) if len(drugs) > 1 else 0
#                 batch_ddi_loss += ddi_score
        
#         return batch_ddi_loss / y_prob.size(0)

#     def wrap_1d_to_3d(raw_list, batch_size):
#         """
#         raw_list: 整体维度是 [code1, code2, ...] 或 []
#         返回: [[raw_list]] * batch_size  （即 [B, 1, M]）
#         """
#         visit = raw_list if isinstance(raw_list, list) else []
#         return [ [visit] for _ in range(batch_size) ]

#     def _load_ddi_adj(self, ddi_pkl_path):
#         """从本地 pkl 文件加载 DDI 邻接矩阵"""
#         if ddi_pkl_path and os.path.exists(ddi_pkl_path):
#             with open(ddi_pkl_path, 'rb') as f:
#                 ddi_adj = pickle.load(f)
#             if not isinstance(ddi_adj, torch.Tensor):
#                 ddi_adj = torch.FloatTensor(ddi_adj)
#             if ddi_adj.shape != (self.vocab_size-2, self.vocab_size-2):
#                 raise ValueError(f"DDI adjacency matrix shape {ddi_adj.shape} does not match vocab_size {self.vocab_size}")
#             return ddi_adj
#         else:
#             # 默认全零矩阵
#             return torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.float)

#     def forward(
#             self,
#             patient_id: List[List[str]],
#             conditions: List[List[List[str]]],  # 需要和dataset保持一致[名字，因为**data]
#             procedures: List[List[List[str]]],
#             drugs_hist: List[List[List[str]]],
#             labels: List[List[str]],  # label
#             **kwargs,
#     ) -> Dict[str, torch.Tensor]:
#         """Forward propagation.
#         Returns:
#             A dictionary with the following keys:
#                 loss: a scalar tensor representing the loss.
#                 y_prob: a tensor of shape [patient, visit, num_labels]
#                     representing the probability of each drug.
#                 y_true: a tensor of shape [patient, visit, num_labels]
#                     representing the ground truth of each drug.
#         """
#         labels_copy = labels  # for case
#         # # patient id
#         # prepare labels
#         if self.mode == "multilabel":
#             labels_index = self.label_tokenizer.batch_encode_2d(
#                 labels, padding=False, truncation=False
#             )  # [[23,32],[1,2,3]]，注意比feature_tokenizer少两位
#             labels_multihot = batch_to_multihot(labels_index, self.label_size)  # [B, num_labels]
#             y_true = labels_multihot.to(self.device).unsqueeze(1)  # [B, 1, num_labels]

#             labels = batch_to_multihot(labels_index, self.label_size)  # tensor, B, Label_size;  # convert to multihot
#             index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
#             for idx, cont in enumerate(labels_index):
#                 # remove redundant labels
#                 cont = list(set(cont))
#                 index_labels[idx, : len(cont)] = cont  # remove padding and unk

#             index_labels = torch.from_numpy(index_labels)  # 类似的！【23，38，39】
#             labels = labels.to(self.device)  # for bce loss
#             index_labels = index_labels.to(self.device)  # for multi label loss

#         # patient id
#         # kwargs['miss_feature'] = torch.tensor(kwargs['miss_feature']).to(self.device)
#         cond_code, cond_emb, condi_mask, condition_vis_emb = self.encode_patient("conditions",
#                                                                           conditions, kwargs['new_conditions'])  # [B,V,M] [B,V,M,D]; [B,V,M], [B,V,D]
#         proc_code, proc_emb, proc_mask, procedure_vis_emb = self.encode_patient("procedures", procedures, kwargs['new_procedures'])
#         drug_code, drug_emb, drug_mask, drug_history_vis_emb = self.encode_patient("drugs",
#                                                                             drugs_hist, kwargs['new_drugs'])  # drug rec的时候不能放drug 1，1，1，1
#         # tianzhi add 这里返回的四个东西分别是 原始编码，原始embedding，mask（用于表示那些是填充，哪些是有意义的），visit-level embedding （补全后合并的数据，都合并了）

#         # 获取患者当前的表示 
#         next_conditions_3d = [[visit] for visit in kwargs['next_conditions']]
#         next_procedures_3d = [[visit] for visit in kwargs['next_procedures']]
        
#         _, next_cond_emb, next_cond_mask, _ = self.encode_patient("conditions", next_conditions_3d,0)  # [B,V,M] [B,V,M,D]; [B,V,M], [B,V,D]
#         _, next_pro_emb, next_pro_mask, _ = self.encode_patient("procedures", next_procedures_3d,0)  

#         current_fusion_emb, current_visit_mask = self.fusion(next_cond_emb, next_pro_emb, next_cond_mask, next_pro_mask)

#         current_repr = self.patient_representation(current_fusion_emb, current_visit_mask) # 患者当前访问表示

#         fusion_emb, visit_mask = self.fusion(cond_emb, proc_emb, condi_mask, proc_mask)  # [B,V,D] 这里是visit-level embedding
       
#         patient_id = self.patient_representation(fusion_emb, visit_mask)
#         historical_repr = patient_id    # 患者历史访问表示

#         alpha = torch.sigmoid(self.alpha)  # 保证 α ∈ (0,1)
#         final_repr = alpha * historical_repr + (1 - alpha) * current_repr

#         # 药物推荐
#         scores = self.DrugRecommendationModel(final_repr)  # [B, num_drugs], [B, 1, num_drugs]
#         y_prob = torch.sigmoid(scores).unsqueeze(1)  # [B, 1, num_drugs]
#         # ipdb.set_trace()
#         # 计算损失
#         loss = self.DrugRecommendationModel.calculate_loss(scores, y_true.squeeze(1))
#         # _, ddi_embeddings = self.gcn.forward()
#         # # 提取药物嵌入
#         # drug_embeddings = ddi_embeddings[self.drug_voc_indices, :].to(self.device)

#         # scores = final_repr @ drug_embeddings.T  # [B, emb_dim] @ [emb_dim, label_size] = [B, label_size]
        
#         # ipdb.set_trace()
#         # print("y_true size:", y_true.size())
#         # print("y_prob size:", y_prob.size())
#         # y_prob = y_prob[:, :, :y_true.size(-1)]
#         # # 计算损失

#         return {
#             "loss": loss,
#             "y_prob": y_prob,
#             "y_true": y_true
#         }
        


