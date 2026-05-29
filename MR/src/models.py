import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
import math
from torch.nn.parameter import Parameter
from collections import Counter, defaultdict
import math
from module import RobustSemanticCalibrator, RobustGlobalRetrievalModule
"""
Our model
"""

####
class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        device = torch.device("cuda:0")

        adj = torch.tensor(adj, dtype=torch.float32).to(device)
        identity_matrix = torch.eye(adj.shape[0]).to(device)
        adj = self.normalize(adj + identity_matrix)
        self.adj = nn.Parameter(adj, requires_grad=False)
        
        
        self.x = nn.Parameter(torch.eye(voc_size).to(device), requires_grad=False)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    # gpu version
    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        mx = mx.to_dense()
        rowsum = mx.sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diagflat(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx
####

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class SafeDrugModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        ddi_adj,
        ehr_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprints,
        average_projection,
        emb_dim=256,
        device=torch.device("cpu:0"),
        graph_branch_init=0.1,
        support_local_weight=0.7,
    ):
        super(SafeDrugModel, self).__init__()
        self.x = nn.Parameter(torch.tensor(0.0))
        self.device = device
        self.scale_factor = math.sqrt(emb_dim)
        graph_branch_init = float(graph_branch_init)
        if not 0.0 <= graph_branch_init <= 1.0:
            raise ValueError("graph_branch_init must be in the closed interval [0, 1]")
        support_local_weight = float(support_local_weight)
        if not 0.0 <= support_local_weight <= 1.0:
            raise ValueError("support_local_weight must be in the closed interval [0, 1]")
        # graph branch is now a fixed runtime weight instead of a trainable logit.
        self.graph_branch_weight = graph_branch_init
        self.support_local_weight = support_local_weight
        self.support_global_weight = 1.0 - support_local_weight

        self.ehr_gcn = GCN(voc_size=ehr_adj.shape[0], emb_dim=emb_dim, adj=ehr_adj)
        self.ddi_gcn = GCN(voc_size=ddi_adj.shape[0], emb_dim=emb_dim, adj=ddi_adj)

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)] 
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(3)]
        )
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(3 * emb_dim, emb_dim))

        self.imputation_weighter = RobustSemanticCalibrator(embed_dim=emb_dim)

        self.global_retrieval = RobustGlobalRetrievalModule(
            view_vocab_sizes={
                "conditions": vocab_size[0],
                "procedures": vocab_size[1],
                "drugs": vocab_size[2],
            },
            k_neighbors=10,
            floor_weight=0.1,
        )
        self.support_floor = self.global_retrieval.floor_weight

        self.mol_score_proj = nn.Linear(vocab_size[2], vocab_size[2])
        self.mol_score_norm = nn.LayerNorm(vocab_size[2])
        self.graph_score_proj = nn.Linear(vocab_size[2], vocab_size[2])
        self.graph_score_norm = nn.LayerNorm(vocab_size[2])

        self.register_buffer("graph_branch_logit", torch.tensor(0.0, dtype=torch.float32))

        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1]) 
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)

        # MPNN global embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))
        
        with torch.no_grad():
            mpnn_emb = MolecularGraphNeuralNetwork(
                N_fingerprints, emb_dim, layer_hidden=2, device=device
            ).forward(self.MPNN_molecule_Set)
            mpnn_emb = torch.mm(
                average_projection.to(device=self.device),
                mpnn_emb.to(device=self.device),
            )
        self.register_buffer("MPNN_emb", mpnn_emb)
        
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()


    def build_retrieval_bank(self, train_data_list):
        structured_bank_codes = []

        for patient in train_data_list:
            for adm in patient:
                if sum(adm[3]) != 0:
                    continue

                structured_bank_codes.append(
                    {
                        "conditions": set(adm[0]),
                        "procedures": set(adm[1]),
                        "drugs": set(adm[2]),
                    }
                )

        self.global_retrieval.build_bank(structured_bank_codes)

    def _compose_support_weight(self, local_weight=None, global_weight=None):
        if local_weight is None and global_weight is None:
            return None
        if local_weight is None:
            combined_support = global_weight
        elif global_weight is None:
            combined_support = local_weight
        else:
            combined_support = (
                self.support_local_weight * local_weight
                + self.support_global_weight * global_weight
            )
        return self.support_floor + (1.0 - self.support_floor) * combined_support

    def forward(self, input):

        # patient health representation
        i1_seq = []
        i2_seq = []
        i3_seq = []

        def mean_embedding(embedding):
            
            if embedding is None or embedding.size(1) == 0:
                 return torch.zeros(1, 1, self.embeddings[0].embedding_dim).to(self.device)
            return embedding.mean(dim=1).unsqueeze(dim=0)

        
        seq_len = len(input)

        for idx, adm in enumerate(input):
            is_target_visit = (idx == seq_len - 1)
            c_emb = self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(0).to(self.device))
            p_emb = self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(0).to(self.device))

            if not is_target_visit:
                d_emb = self.embeddings[2](torch.LongTensor(adm[2]).unsqueeze(0).to(self.device))
                current_drug_codes = adm[2]
            else:
                d_emb = None
                current_drug_codes = []

            input_embeddings_dict = {
                'conditions': c_emb,
                'procedures': p_emb,
                'drugs': d_emb  # 如果是最后一次访问，这里是 None
            }

            input_raw_codes_dict = {
                'conditions': adm[0],
                'procedures': adm[1],
                'drugs': current_drug_codes
            }

            local_weights_dict = self.imputation_weighter(
                input_embeddings_dict,
                adm[3]
            )

            global_weights_dict = self.global_retrieval(
                input_embeddings_dict,
                adm[3],
                input_raw_codes_dict
            )

            final_embeddings_dict = input_embeddings_dict.copy()
            for view_name in ['conditions', 'procedures', 'drugs']:
                if final_embeddings_dict.get(view_name) is None:
                    continue

                fused_weight = self._compose_support_weight(
                    local_weight=local_weights_dict.get(view_name),
                    global_weight=global_weights_dict.get(view_name),
                )
                if fused_weight is not None:
                    final_embeddings_dict[view_name] = final_embeddings_dict[view_name] * fused_weight

            i1 = self.dropout(final_embeddings_dict['conditions'])
            i1 = mean_embedding(i1)
            i1_seq.append(i1)

            i2 = self.dropout(final_embeddings_dict['procedures'])
            i2 = mean_embedding(i2)
            i2_seq.append(i2)

            if final_embeddings_dict['drugs'] is not None:
                i3 = self.dropout(final_embeddings_dict['drugs'])
            else:
                i3 = torch.zeros_like(i1)
            i3 = mean_embedding(i3)
            i3_seq.append(i3)


        i1_seq = torch.cat(i1_seq, dim=1)  
        i2_seq = torch.cat(i2_seq, dim=1)  
        i3_seq = torch.cat(i3_seq, dim=1)   

        o1, h1 = self.encoders[0](i1_seq)
        o2, h2 = self.encoders[1](i2_seq)
        o3, _ = self.encoders[2](i3_seq)

        patient_features = torch.cat([o1, o2, o3], dim=-1).squeeze(dim=0)
        query = self.query(patient_features)[-1:, :]

        mol_match_score = torch.mm(query, self.MPNN_emb.t())
        mol_match_score = mol_match_score / self.scale_factor
        mol_match_score = F.sigmoid(mol_match_score)
        mol_semantic = self.mol_score_norm(
            mol_match_score + self.mol_score_proj(mol_match_score)
        )

        bipartite_emb = self.bipartite_output(
            F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t()
        )
        molecular_score = torch.mul(bipartite_emb, mol_semantic)

        ehr_meds = self.ehr_gcn()  # (med_count , dim)
        ddi_meds = self.ddi_gcn()
        self.ehr_ddi_resp = (ehr_meds - ddi_meds * torch.sigmoid(self.x)).to(self.device)

        graph_match_score = torch.mm(query, self.ehr_ddi_resp.t())
        graph_match_score = graph_match_score / self.scale_factor
        graph_match_score = F.sigmoid(graph_match_score)
        graph_score = self.graph_score_norm(
            graph_match_score + self.graph_score_proj(graph_match_score)
        )
        result = molecular_score + self.graph_branch_weight * graph_score

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
