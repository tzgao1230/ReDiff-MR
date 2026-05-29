# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : note_pre.py
# Time       ：12/12/2024 4:32 pm
# Author     ：XXXX
# version    ：python 
# Description：
"""
import os
import re
import string
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import BioGptTokenizer, BioGptModel
from utils import save_pickle, load_pickle



# Remove symbols and line breaks from text
def remove_symbol(text):
    text = text.replace('\n', '')

    punctuation_string = string.punctuation
    for i in punctuation_string:
        text = text.replace(i, '')

    return text


# extract Brief Hospital Course in the discharge summary
def extract_BHC(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern1 = re.compile(r"brief hospital course:(.*?)medications on admission", re.DOTALL)
    pattern2 = re.compile(r"brief Hospital Course:(.*?)discharge medications", re.DOTALL)

    if "brief hospital course:" in text:
        if re.search(pattern1, text):
            match = re.search(pattern1, text).group(1).strip()
        elif re.search(pattern2, text):
            match = re.search(pattern2, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Chief Complaint in the discharge summary
def extract_CC(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"chief complaint:(.*?)major surgical or invasive procedure", re.DOTALL)

    if "chief complaint:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Past Medical History in the discharge summary
def extract_PMH(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"past medical history:(.*?)social history", re.DOTALL)

    if "past medical history:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# extract Medications on Admission in the discharge summary
def extract_MA(text):
    text = text.lower()

    # using regular expression to extract the content
    pattern = re.compile(r"medications on admission:(.*?)discharge medications", re.DOTALL)

    if "medications on admission:" in text:
        if re.search(pattern, text):
            match = re.search(pattern, text).group(1).strip()
        else:
            match = None
    else:
        match = None

    if match is not None:
        match = remove_symbol(match)

    return match


# Extract required data from raw MIMIC-NOTE
def parse_note(note_path, output_path):
    new_df = pd.read_csv(os.path.join(note_path, 'note/discharge.csv'))

    # Extract Brief Hospital Course part as the overall admission summary (For DRG task)
    new_df['brief_hospital_course'] = new_df['text'].apply(extract_BHC) # 最后我看只用了这个。

    # Extract Past Medical History  (For 6 tasks)
    new_df['past_medical_history'] = new_df['text'].apply(extract_PMH)

    # Extract Chief Complaint and Medications on Admission
    # new_df['chief_complaint'] = new_df['text'].apply(extract_CC)
    # new_df['medications_on_admission'] = new_df['text'].apply(extract_MA)

    new_df.drop(['text', 'note_id', 'note_type', 'note_seq'], axis=1, inplace=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    new_df.rename(columns={'subject_id': 'patient_id', 'hadm_id': 'visit_id'}, inplace=True)
    new_df['past_medical_history'] = new_df['past_medical_history'].fillna("Unknown")
    new_df['brief_hospital_course'] = new_df['brief_hospital_course'].fillna("Unknown")


    new_df.to_csv(os.path.join(output_path, 'note_all.csv'), index=False)

    return new_df




def run_pretrain_plm(model_name, config):
    device = 'cuda:' + config['GPU'] if config['USE_CUDA'] else 'cpu'
    if model_name == 'Sap-BERT':
        tokenizer = AutoTokenizer.from_pretrained("/home/user4/tianzhi/model/cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model = AutoModel.from_pretrained("/home/user4/tianzhi/model/cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)
    elif model_name == 'BioGPT':
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt", model_max_length=384)
        model = BioGptModel.from_pretrained("microsoft/biogpt").to(device)
        # text = "Replace me by any text you'd like."
        # encoded_input = tokenizer(text, return_tensors='pt')
        # output = model(**encoded_input)
    elif model_name == 'BioMistral':
        tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B",model_max_length=512)# 需要torch 1.6
        model = AutoModel.from_pretrained("BioMistral/BioMistral-7B").to(device)
    elif model_name == 'Clinical-BERT':
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT", model_max_length=512)
        model = AutoModel.from_pretrained("medicalai/ClinicalBERT").to(device)
    elif model_name == 'Clinical-T5':
        # 这里用于下载的seq2seq模型。
        pass

    else:
        raise ValueError("Model not supported!")
    return model, tokenizer


def get_embedding(model, tokenizer, all_text, plm_model_name):
    print("Extract embeddings...")
    bs = 4  # batch size during inference
    all_embs = []
    if plm_model_name == 'Sap-BERT':
        for i in tqdm(np.arange(0, len(all_text), bs)):
            try:
                toks = tokenizer.batch_encode_plus(all_text[i:i + bs],
                                                   padding="max_length",
                                                   max_length=25,
                                                   truncation=True,
                                                   return_tensors="pt")
            except:
                updated_list = ['Unknown' if (isinstance(v, float) and math.isnan(v)) or v != v else v for v in
                                all_text[i:i + bs]]

                toks = tokenizer.batch_encode_plus(updated_list,
                                                   padding="max_length",
                                                   max_length=25,

                                                   truncation=True,
                                                   return_tensors="pt")

            toks_cuda = {}
            for k, v in toks.items():
                toks_cuda[k] = v.to(model.device)
            cls_rep = model(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
            all_embs.append(cls_rep.cpu().detach())
    elif plm_model_name in ['BioGPT', 'Clinical-BERT', 'BioMistral']:
        for i in tqdm(np.arange(0, len(all_text), bs)):
            try:
                toks = tokenizer(all_text[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=50) # 居然最大的只有这么大；
            except:
                updated_list = ['Unknown' if (isinstance(v, float) and math.isnan(v)) or v != v else v for v in
                                all_text[i:i + bs]]
                toks = tokenizer(updated_list, return_tensors="pt", padding=True, truncation=True)

            toks_cuda = {}
            for k, v in toks.items():
                toks_cuda[k] = v.to(model.device)
                # print(v.shape)
            embeddings = model(**toks_cuda).last_hidden_state  # use CLS representation as the embedding
            sentence_embeddings = torch.mean(embeddings, dim=1)
            all_embs.append(sentence_embeddings.cpu().detach())

    all_embs = torch.cat(all_embs, dim=0)
    return all_embs



def fast_index(note_dict):
    # keys = np.array(list(note_dict.keys()))  # [(patient_id, visit_id), ...]
    # values = np.array(list(note_dict.values()))  # [emb, ...]
    return note_dict# keys, values

def get_note(config, data_path, col_num='past_medical_history'):

    root_to = '/home/qluai/hgc/tianzhi/MMHealth/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])

    emb_dir = root_to+ 'note_emb_cli.pkl'

    if os.path.exists(emb_dir):
        print("Exising Note embeddings found!")
        print("Target dir ", root_to)
        dic = load_pickle(emb_dir)
        return fast_index(dic)
    else:
        print("Target dir ", root_to)
        parse_dir = root_to + 'note_all.csv'
        if not os.path.exists(parse_dir):
            print("Parsing note data...")
            dataframe = parse_note(data_path, root_to)#######################################
        else:
            print("Loading note data...")
            dataframe = pd.read_csv(parse_dir, index_col=None)
            

        all_text = dataframe['past_medical_history'].values # array
        model, tokenizer = run_pretrain_plm(config['PLM'], config)
        all_embs = get_embedding(model, tokenizer, all_text, config['PLM'])
        dataframe['note_emb'] = all_embs.cpu().numpy().tolist()
        dic = dataframe[['patient_id', 'visit_id', 'note_emb']]
        # note_dict = {str(row['patient_id']) + '_' + str(row['visit_id']): row['note_emb'] for _, row in dic.iterrows()}
        dic.set_index(['patient_id', 'visit_id'], inplace=True)
        note_dict = dic
        save_pickle(note_dict, emb_dir)
        print('save note done!')
        return fast_index(dic)



if __name__ == '__main__':
    # 测试 note抽取
    # wget -r -N -c -np --user XXXX-3 --ask-password https://physionet.org/files/mimiciv/3.1/
    # wget -r -N -c -np --user XXXX-3 --ask-password https://physionet.org/files/mimiciv/2.2/
    from config import config
    note_dic = get_note(config, "/home/qluai/hgc/tianzhi/data/physionet.org/files/mimiciv/")
