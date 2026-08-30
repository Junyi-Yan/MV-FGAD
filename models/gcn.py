import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import *
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from models.message_op.laplacian_graph_op import LaplacianGraphOp
from models.message_op.concat_message_op import ConcatMessageOp

class ChebNetConv(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(ChebNetConv, self).__init__()

        self.K = k
        self.linear = nn.Linear(in_features * k, out_features)

    def forward(self, x, laplacian):
        x = self.__transform_to_chebyshev(x, laplacian)
        x = self.linear(x)
        return x

    def __transform_to_chebyshev(self, x, laplacian):
        cheb_x = x.unsqueeze(2)
        x0 = x

        if self.K > 1:
            x1 = torch.mm(laplacian, x0)
            cheb_x = torch.cat((cheb_x, x1.unsqueeze(2)), 2)
            for _ in range(2, self.K):
                x2 = 2 * torch.mm(laplacian, x1) - x0
                cheb_x = torch.cat((cheb_x, x2.unsqueeze(2)), 2)
                x0, x1 = x1, x2

        cheb_x = cheb_x.reshape([x.shape[0], -1])
        return cheb_x


def getre_scale(emb):
    emb_softmax = nn.Softmax(dim=1)(emb)
    re = torch.mm(emb_softmax, emb_softmax.transpose(0,1))
    re_self = torch.unsqueeze(torch.diag(re),1)
    scaling = torch.mm(re_self, torch.transpose(re_self, 0, 1))
    re = re / torch.max(torch.sqrt(scaling),1e-9*torch.ones_like(scaling))
    re = re - torch.diag(torch.diag(re))
    return re

def add_diag(re_matrix, device):
    re_diag = torch.diag(re_matrix)
    re_diag_matrix = torch.diag_embed(re_diag)
    re = re_matrix - re_diag_matrix
    re = re_matrix + torch.eye(re_matrix.shape[0]).to(device)
    return re


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        return output

class GCN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(feat_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.post_graph_op = None
        self.use_graph_op = True
        self.pre_graph_op = None

    def preprocess(self, adj, feature):
        self.pre_msg_learnable = False
        self.processed_feature = feature
        self.adj = sparse_mx_to_torch_sparse_tensor(adj_to_symmetric_norm(adj, r=0.5))

    def model_forward(self, idx, device):
        x = self.processed_feature.to(device)
        A = self.adj.to(device)
        x = self.conv1(x, A)
        x = self.relu(x)
        x = self.dropout(x)
        x2 = self.conv2(x, A)
        emb = x2 + self.proj(x)
        anomaly_score = mahalanobis_distance_normalized(emb)

        return emb[idx], anomaly_score[idx]


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        N = x.size(0)
        residual = self.shortcut(x)
        Q = self.q_proj(x).view(N, self.num_heads, -1)
        K = self.k_proj(x).view(N, self.num_heads, -1)
        V = self.v_proj(x).view(N, self.num_heads, -1)

        scores = torch.matmul(Q.transpose(0, 1), K.transpose(0, 1).transpose(-1, -2))
        scores = scores / (self.out_dim // self.num_heads) ** 0.5
        adj_dense = adj.to(x.device)
        scores = scores.masked_fill(adj_dense.unsqueeze(0) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)

        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, V.transpose(0, 1)).transpose(0, 1).contiguous()
        out = out.view(N, -1)
        x = self.norm1(residual + out)
        x = self.norm2(x + self.ffn(x))
        return x


class GraphTransformer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, num_heads=4, dropout=0.5):
        super(GraphTransformer, self).__init__()
        self.layer1 = GraphTransformerLayer(feat_dim, hidden_dim, num_heads, dropout)
        self.layer2 = GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def preprocess(self, adj, feature):
        self.pre_msg_learnable = False
        self.processed_feature = feature

        if sp.issparse(adj):
            adj = adj + sp.eye(adj.shape[0])
            adj = torch.FloatTensor(adj.toarray())
        elif not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float32)
            adj = adj + torch.eye(adj.size(0))
        else:
            adj = adj + torch.eye(adj.size(0)).to(adj.device)

        self.adj = adj

    def model_forward(self, idx, device):
        x = self.processed_feature.to(device)
        A = self.adj.to(device)
        x = self.layer1(x, A)
        x = self.dropout(x)
        x_hidden = self.layer2(x, A)
        emb = self.out_proj(x_hidden) + self.proj(x)
        anomaly_score = mahalanobis_distance_normalized(emb)
        return emb[idx], anomaly_score[idx]

class ChebNet(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.1, bn=False, ln=False, k=2):
        super(ChebNet, self).__init__()

        self.use_graph_op = True
        self.pre_graph_op = None

        self.conv1 = ChebNetConv(feat_dim, hidden_dim, k)
        self.conv2 = ChebNetConv(hidden_dim, output_dim, k)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.post_graph_op = None

    def preprocess(self, adj, feature):
        self.pre_msg_learnable = False
        self.processed_feature = feature
        
        adj = adj_to_symmetric_norm(adj, r=0.5)

        self.adj = sparse_mx_to_torch_sparse_tensor(adj)

    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def forward(self, idx, device):
        processed_feature = None
        if self.pre_msg_learnable is False:
            processed_feature = self.processed_feature.to(device)
        else:
            transferred_feat_list = [feat.to(
                device) for feat in self.processed_feat_list]
            processed_feature = self.pre_msg_op.aggregate(
                transferred_feat_list)

        self.adj = self.adj.to(device)
        x = processed_feature
        x = self.conv1(x, self.adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.adj)

        return x[idx]

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)
        return output