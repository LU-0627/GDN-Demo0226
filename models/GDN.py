import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, edge_weight=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(
            x,
            edge_index,
            embedding,
            edge_weight=edge_weight,
            return_attention_weights=True,
        )
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


class GDN(nn.Module):
    def __init__(
        self,
        edge_index_sets,
        node_num,
        dim=64,
        out_layer_inter_dim=256,
        input_dim=10,
        out_layer_num=1,
        topk=20,
        moe_num=4,
        low_rank_dim=8,
        route_topk=2,
        gumbel_tau=1.0,
    ):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]


        edge_set_num = len(edge_index_sets)
        embed_dim = dim
        hidden_dim = dim * edge_set_num
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(hidden_dim)

        self.node_num = node_num
        self.embed_dim = embed_dim

        self.moe_num = max(2, int(moe_num))
        self.low_rank_dim = max(1, int(low_rank_dim))
        self.route_topk = max(2, int(route_topk))
        self.sparse_topk = int(topk)
        self.gumbel_tau = float(gumbel_tau)

        self.e_base = nn.Parameter(torch.empty(node_num, embed_dim))
        self.low_rank_u = nn.Parameter(torch.empty(self.moe_num, node_num, self.low_rank_dim))
        self.low_rank_v = nn.Parameter(torch.empty(self.moe_num, self.low_rank_dim, embed_dim))

        self.cond_encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.router = nn.Linear(hidden_dim, self.moe_num, bias=True)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        # Condition sensing: subsystem-level local-sensitive attention readout
        # W_h: [H, H], w_attn: [H, 1], where H = dim * edge_set_num
        self.cond_attn_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.cond_attn_vec = nn.Linear(hidden_dim, 1, bias=False)
        self.cond_attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.e_base, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.low_rank_u)
        nn.init.xavier_uniform_(self.low_rank_v)

    def _sample_gumbel(self, shape, device):
        uniform = torch.rand(shape, device=device).clamp_(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def _batch_sparse_graph(self, mixed_embed, topk_num):
        # mixed_embed: [B, N, d]
        batch_size, num_nodes, _ = mixed_embed.shape
        use_k = max(1, min(topk_num, num_nodes))
        device = mixed_embed.device

        # 1) Batched similarity by BMM: [B, N, d] @ [B, d, N] -> [B, N, N]
        scores = torch.bmm(mixed_embed, mixed_embed.transpose(1, 2))  # shape: [B, N, N]
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)

        # 2) Batched TopK on node dimension
        topk_vals, topk_idx = torch.topk(scores, k=use_k, dim=-1)  # shape: [B, N, K], [B, N, K]
        topk_w = F.softmax(topk_vals, dim=-1)  # shape: [B, N, K]

        # 3) Build destination indices
        dst_idx = torch.arange(num_nodes, device=device).view(1, num_nodes, 1).expand(batch_size, num_nodes, use_k)  # shape: [B, N, K]

        # 4) Batch offset
        batch_offset = (torch.arange(batch_size, device=device) * num_nodes).view(batch_size, 1, 1)  # shape: [B, 1, 1]

        # 5) Flatten to sparse edge format
        src = (topk_idx + batch_offset).flatten()  # shape: [B*N*K]
        dst = (dst_idx + batch_offset).flatten()  # shape: [B*N*K]

        batch_edge_index = torch.stack((src, dst), dim=0).long()  # shape: [2, B*N*K]
        batch_edge_weight = topk_w.flatten()  # shape: [B*N*K]

        return batch_edge_index, batch_edge_weight


    def forward(self, data, org_edge_index):

        x = data.clone().detach()

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        # Module-1 feature extraction for routing input
        # h_{i,t}: [B, N, H]
        h_it = self.cond_encoder(data)

        # e_{i,t} = w_attn^T LeakyReLU(W_h h_{i,t})
        # attn_hidden: [B, N, H]
        attn_hidden = self.cond_attn_act(self.cond_attn_proj(h_it))
        # e_{i,t}: [B, N]
        e_it = self.cond_attn_vec(attn_hidden).squeeze(-1)
        e_it = torch.nan_to_num(e_it, nan=0.0, posinf=1e4, neginf=-1e4)

        # beta_{i,t} = Softmax(e_{i,t}) over node dimension
        # beta_{i,t}: [B, N]
        beta_it = F.softmax(e_it, dim=1)

        # h_{sys,t} = sum_i beta_{i,t} h_{i,t}
        # h_{sys,t}: [B, H]
        h_sys_t = torch.sum(beta_it.unsqueeze(-1) * h_it, dim=1)

        # ---------------- Sparse MoE Routing (Module-2) ----------------
        # (a) Routing logits z_t from h_{sys,t}
        # z_t: [B, M]
        z_t = self.router(h_sys_t)

        # (b) pi_soft = Softmax((z_t + g_t) / tau)
        # g_t: [B, M]
        g_t = self._sample_gumbel(z_t.shape, device)
        # pi_soft: [B, M]
        pi_soft = F.softmax((z_t + g_t) / max(self.gumbel_tau, 1e-6), dim=-1)

        # (c) Top-2 hard routing
        # topk_idx: [B, 2], topk_val: [B, 2]
        topk_val, topk_idx = torch.topk(pi_soft, k=self.route_topk, dim=-1)
        # pi_hard: [B, M]
        pi_hard = torch.zeros_like(pi_soft)
        pi_hard.scatter_(1, topk_idx, topk_val)
        pi_hard = pi_hard / pi_hard.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # (d) Straight-Through estimator
        # pi_t: [B, M]
        pi_t = (pi_hard - pi_soft).detach() + pi_soft

        # (e) Low-rank prototype embeddings
        # low_rank_delta: [M, N, d]
        low_rank_delta = torch.matmul(self.low_rank_u, self.low_rank_v)
        # proto_embed: [M, N, d]
        proto_embed = self.e_base.unsqueeze(0) + low_rank_delta

        # (f) Top-2 weighted merge (per sample)
        # mixed_embed: [B, N, d]
        mixed_embed = torch.einsum('bm,mnd->bnd', pi_t, proto_embed)
        mixed_embed = torch.nan_to_num(mixed_embed, nan=0.0, posinf=1e4, neginf=-1e4)

        # (g) Post-merge re-sparsification with Top-K'
        # batch_edge_index: [2, E], batch_edge_weight: [E]
        batch_edge_index, batch_edge_weight = self._batch_sparse_graph(mixed_embed, self.sparse_topk)
        batch_edge_index = batch_edge_index.to(device)
        batch_edge_weight = batch_edge_weight.to(device)

        # embedding for graph attention term
        # all_embeddings: [B*N, d]
        all_embeddings = mixed_embed.reshape(batch_num * node_num, -1)

        gcn_outs = []
        for i in range(len(self.gnn_layers)):
            gcn_out = self.gnn_layers[i](
                x,
                batch_edge_index,
                embedding=all_embeddings,
                edge_weight=batch_edge_weight,
                node_num=node_num * batch_num,
            )
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0,node_num).to(device)
        node_embed = self.embedding(indexes)
        if x.shape[-1] != node_embed.shape[-1]:
            repeat_factor = x.shape[-1] // node_embed.shape[-1]
            node_embed = node_embed.repeat(1, repeat_factor)
        out = torch.mul(x, node_embed.unsqueeze(0))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
   

        return out, h_sys_t, pi_soft
