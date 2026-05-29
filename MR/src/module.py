import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
import math
from torch.nn.parameter import Parameter
import ipdb
from collections import Counter
from einops import rearrange, repeat, einsum
from typing import Union
from collections import Counter, defaultdict
import math


class RobustSemanticCalibrator(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, attn_topk=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_topk = attn_topk
        self.view_names = ['conditions', 'procedures', 'drugs']
        self.view_type_embeds = nn.Parameter(torch.randn(len(self.view_names), embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        hidden_dim = max(num_heads * 2, 8)
        self.confidence_layer = nn.Sequential(
            nn.Linear(num_heads * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings_dict, incomplete_flag):
        real_list, imputed_info = [], []
        for i, name in enumerate(self.view_names):
            emb = embeddings_dict.get(name)
            if emb is None:
                continue

            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            if emb.size(1) == 0:
                continue
            emb_with_type = emb + self.view_type_embeds[i].view(1, 1, -1)
            if incomplete_flag[i] == 0:
                real_list.append(emb_with_type)
            else:
                imputed_info.append({'name': name, 'raw': emb, 'typed': emb_with_type})
        if not real_list or not imputed_info:
            return {}
        context_emb = torch.cat(real_list, dim=1)
        support_weights = {}

        for item in imputed_info:
            attended_context, attn_weights = self.cross_attn(
                query=item['typed'], key=context_emb, value=context_emb, average_attn_weights=False
            )
            # attn_weights: [B, Heads, Seq_imputed, Seq_context]
            topk_k = min(self.attn_topk, attn_weights.size(-1))
            topk_attn = torch.topk(attn_weights, k=topk_k, dim=-1).values
            attn_topk_mean = topk_attn.mean(dim=-1).transpose(1, 2)

            attn_probs = attn_weights.clamp_min(1e-12)
            attn_entropy = -(attn_probs * attn_probs.log()).sum(dim=-1)
            if attn_weights.size(-1) > 1:
                attn_entropy = attn_entropy / math.log(attn_weights.size(-1))
            else:
                attn_entropy = torch.zeros_like(attn_entropy)
            attn_entropy = attn_entropy.transpose(1, 2)

            raw_norm = F.normalize(item['raw'], dim=-1)
            ctx_norm = F.normalize(attended_context, dim=-1)
            alignment = F.cosine_similarity(raw_norm, ctx_norm, dim=-1).unsqueeze(-1)

            confidence_features = torch.cat(
                [attn_topk_mean, attn_entropy, alignment], dim=-1
            )
            local_support = torch.sigmoid(self.confidence_layer(confidence_features))
            support_weights[item['name']] = local_support.to(
                device=item['raw'].device,
                dtype=item['raw'].dtype,
            )
        return support_weights



class RobustGlobalRetrievalModule(nn.Module):
    def __init__(
        self,
        view_vocab_sizes,
        k_neighbors=15,
        floor_weight=0.1,
        view_weights=None,
    ):
        super().__init__()
        self.k = k_neighbors
        self.floor_weight = floor_weight
        self.view_weights = view_weights or {
            "conditions": 1.0,
            "procedures": 1.0,
            "drugs": 1.0,
        }
        self.view_names = ["conditions", "procedures", "drugs"]
        self.view_vocab_sizes = view_vocab_sizes
        self.bank_visits = []
        self.is_ready = False

        for view_name in self.view_names:
            vocab_size = self.view_vocab_sizes[view_name]
            self.register_buffer(
                f"{view_name}_bank",
                torch.empty((0, vocab_size), dtype=torch.float32),
            )
            self.register_buffer(
                f"{view_name}_bank_counts",
                torch.empty(0, dtype=torch.float32),
            )

    def _get_bank_tensor(self, view_name):
        return getattr(self, f"{view_name}_bank")

    def _get_bank_counts(self, view_name):
        return getattr(self, f"{view_name}_bank_counts")

    def _build_multihot_bank(self, bank_visits):
        bank_size = len(bank_visits)
        device = self._get_bank_tensor("conditions").device

        for view_name in self.view_names:
            vocab_size = self.view_vocab_sizes[view_name]
            bank_tensor = torch.zeros(
                (bank_size, vocab_size), device=device, dtype=torch.float32
            )
            bank_counts = torch.zeros(bank_size, device=device, dtype=torch.float32)

            row_indices = []
            col_indices = []
            for row_idx, visit in enumerate(bank_visits):
                codes = visit[view_name]
                if not codes:
                    continue
                bank_counts[row_idx] = float(len(codes))
                row_indices.extend([row_idx] * len(codes))
                col_indices.extend(codes)

            if row_indices:
                row_tensor = torch.tensor(row_indices, device=device, dtype=torch.long)
                col_tensor = torch.tensor(col_indices, device=device, dtype=torch.long)
                bank_tensor[row_tensor, col_tensor] = 1.0

            setattr(self, f"{view_name}_bank", bank_tensor)
            setattr(self, f"{view_name}_bank_counts", bank_counts)

    def build_bank(self, bank_visits):
        self.bank_visits = bank_visits
        if not bank_visits:
            for view_name in self.view_names:
                vocab_size = self.view_vocab_sizes[view_name]
                setattr(
                    self,
                    f"{view_name}_bank",
                    torch.empty((0, vocab_size), device=self._get_bank_tensor(view_name).device, dtype=torch.float32),
                )
                setattr(
                    self,
                    f"{view_name}_bank_counts",
                    torch.empty(0, device=self._get_bank_tensor(view_name).device, dtype=torch.float32),
                )
            self.is_ready = False
            return

        self._build_multihot_bank(bank_visits)
        self.is_ready = True

    def _multi_hot_query(self, codes, view_name):
        bank_tensor = self._get_bank_tensor(view_name)
        query_tensor = torch.zeros(
            self.view_vocab_sizes[view_name],
            device=bank_tensor.device,
            dtype=bank_tensor.dtype,
        )
        if not codes:
            return query_tensor, 0.0

        code_tensor = torch.as_tensor(codes, device=bank_tensor.device, dtype=torch.long)
        code_tensor = torch.unique(code_tensor)
        query_tensor[code_tensor] = 1.0
        return query_tensor, float(code_tensor.numel())

    def forward(self, current_embeddings_dict, incomplete_flag, current_raw_codes):
        
        if not self.is_ready or not self.bank_visits:
            return {}

        view_names = ['conditions', 'procedures', 'drugs']
        
        query_views = []
        imputed_views = []
        
        for i, name in enumerate(view_names):
            if incomplete_flag[i] == 1:
                imputed_views.append(name)
            else:
                if current_raw_codes.get(name):
                    query_views.append(name)

        if not query_views or not imputed_views: 
            return {}

        combined_scores = None
        total_weight = 0.0

        for view_name in query_views:
            query_codes = current_raw_codes.get(view_name, [])
            query_multi_hot, query_count = self._multi_hot_query(query_codes, view_name)
            if query_count == 0.0:
                continue

            bank_tensor = self._get_bank_tensor(view_name)
            bank_counts = self._get_bank_counts(view_name)
            intersections = torch.matmul(bank_tensor, query_multi_hot)
            unions = bank_counts + query_count - intersections
            view_scores = torch.where(
                unions > 0,
                intersections / unions,
                torch.zeros_like(intersections),
            )

            view_weight = self.view_weights.get(view_name, 1.0)
            combined_scores = (
                view_scores * view_weight
                if combined_scores is None
                else combined_scores + view_scores * view_weight
            )
            total_weight += view_weight

        if combined_scores is None or total_weight == 0.0:
            return {}

        combined_scores = combined_scores / total_weight
        topk_k = min(self.k, combined_scores.size(0))
        topk_scores, topk_indices = torch.topk(combined_scores, k=topk_k, largest=True)
        total_neighbor_score = topk_scores.sum()

        global_weights_dict = {}
        
        for target_view in imputed_views:
            target_codes = current_raw_codes.get(target_view, [])
            if not target_codes: 
                continue

            target_bank = self._get_bank_tensor(target_view)[topk_indices]
            positive_mask = topk_scores > 0

            if positive_mask.any():
                vote_scores = torch.matmul(
                    topk_scores[positive_mask], target_bank[positive_mask]
                )
            else:
                vote_scores = torch.zeros(
                    self.view_vocab_sizes[target_view],
                    device=target_bank.device,
                    dtype=target_bank.dtype,
                )

            target_code_tensor = torch.as_tensor(
                target_codes, device=target_bank.device, dtype=torch.long
            )
            if total_neighbor_score.item() > 0.0:
                support_ratio = vote_scores[target_code_tensor] / total_neighbor_score
            else:
                support_ratio = torch.zeros(
                    target_code_tensor.numel(),
                    device=target_bank.device,
                    dtype=target_bank.dtype,
                )

            weight_tensor = support_ratio.to(
                device=current_embeddings_dict[target_view].device,
                dtype=current_embeddings_dict[target_view].dtype,
            ).view(1, -1, 1)
            global_weights_dict[target_view] = weight_tensor

        return global_weights_dict

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
