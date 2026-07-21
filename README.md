# ReDiff-MR

ReDiff-MR: Coarse-to-Fine Diffusion Imputation with Consistency Calibration for Safe Medication Recommendation.

This project contains two parts:

- `DiffImp`: missing-view imputation for incomplete longitudinal EHR records.
- `MR`: medication recommendation.


## Environment

The code is written in Python. A typical environment includes:

```python
python>=3.9
torch
numpy
pandas
scikit-learn
scipy
tqdm
dill
einops
rdkit
dnc
transformers
```

Install the Python dependencies according to your local CUDA/PyTorch environment. For example:

```bash
pip install numpy pandas scikit-learn scipy tqdm dill einops transformers
```

Install `torch`, `rdkit`, and `dnc` with versions compatible with your system.


## Project Structure

```text
ReDiff-MR/
├── DiffImp/
│   └── src/
│       ├── main_unify.py
│       ├── main_diff.py
│       ├── train_dffs.py
│       ├── eval_dffs.py
│       ├── config.py
│       └── diffuser/
│
└── MR/
    ├── src/
    │   ├── main.py
    │   ├── models.py
    │   ├── module.py
    │   ├── layers.py
    │   └── util.py
    └── data/
        └── mimic-iii/
```


## Prepare the Dataset

The medication recommendation task uses the processed MIMIC-III files under:

```text
MR/data/mimic-iii/
```

Required files:

```text
records_final.pkl
voc_final.pkl
ehr_adj_final.pkl
ddi_A_final.pkl
ddi_mask_H.pkl
atc3toSMILES.pkl
```

File descriptions:

- `records_final.pkl`: longitudinal patient visit records.
- `voc_final.pkl`: diagnosis, procedure, and medication vocabularies.
- `ehr_adj_final.pkl`: EHR medication co-occurrence graph.
- `ddi_A_final.pkl`: drug-drug interaction graph.
- `ddi_mask_H.pkl`: DDI mask matrix.
- `atc3toSMILES.pkl`: ATC-to-SMILES mapping for molecular graph construction.


## Train or Test

Run the medication recommendation model:

```bash
cd MR/src
python main.py
```

Common arguments:

```bash
python main.py --lr 5e-4 --target_ddi 0.06 --dim 256 --cuda 0
```

Argument descriptions:

- `--lr`: learning rate.
- `--target_ddi`: target DDI rate.
- `--kp`: coefficient for DDI penalty adjustment.
- `--dim`: embedding dimension.
- `--cuda`: CUDA device index.
- `--graph_branch_init`: fusion coefficient for the EHR-DDI graph branch.

To run in test mode:

```bash
cd MR/src
python main.py --Test --resume_path path/to/model
```
