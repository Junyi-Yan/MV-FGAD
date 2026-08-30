import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args
from models.message_op.concat_message_op import ConcatMessageOp
from models.message_op.laplacian_graph_op import LaplacianGraphOp
from utils import mahalanobis_distance_normalized


class MyModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, hidden_dim, output_dim, r=0.5):
        super().__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = args.alpha
        self.pre_graph_op = LaplacianGraphOp(prop_steps=self.prop_steps, r=r)
        self.pre_msg_op = ConcatMessageOp(start=0, end=self.prop_steps + 1)

    def hete_init(self, device):
        self.hete_model = HetePropagateModel(num_layers=3,
            feat_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            prop_steps=self.prop_steps,
            dropout=0.5,
            bn=False,
            ln=False).to(device)

        self.total_trainable_params = round(
            sum(p.numel() for p in self.hete_model.parameters() if p.requires_grad) / 1000000, 3)

    def non_para_lp(self, subgraph, nodes_embedding, x, device):
        self.nodes_embedding = nodes_embedding.detach()
        self.ori_feature = x.detach()
        self.hete_init(device)


    def preprocess(self, adj, device):
        similarity = getre_scale(self.nodes_embedding)
        self.universal_re = (similarity > 0.9).to(similarity.dtype)
        edge_u, edge_v = torch.where(self.universal_re != 0)
        edge_u = edge_u.detach().cpu().numpy()
        edge_v = edge_v.detach().cpu().numpy()
        universal_re_smooth_adj = sp.coo_matrix(
            (np.ones(len(edge_u), dtype=np.float32), (edge_u, edge_v)),
            shape=(self.nodes_embedding.shape[0], self.nodes_embedding.shape[0]),
        )
        self.adj = self.alpha * adj + (1 - self.alpha) * universal_re_smooth_adj
        self.adj = self.adj.tocoo()

        self.adj = self.adj.tocsr()

        processed_feat_list = self.pre_graph_op.propagate(self.adj, self.ori_feature)
        self.smoothed_feature = self.pre_msg_op.aggregate(processed_feat_list).to(device).detach()

    def hete_forward(self, device, is_train=None):
        if is_train is None:
            is_train = self.training
        local_ori_emb, local_smooth_emb, local_message_propagation, anomaly_score = self.hete_model(
            self.ori_feature,
            self.smoothed_feature,
            self.universal_re,
            device,
            is_train=is_train,
        )
        return local_ori_emb, local_smooth_emb, local_message_propagation, anomaly_score


class HetePropagateLayer(nn.Module):
    def __init__(self, feat_dim, output_dim, prop_steps, hidden_dim, num_layers, dropout=0.5, bn=False, ln=False):
        super().__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.prop_steps = prop_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.bn = bn
        self.ln = ln
        self.beta = args.beta

        self.lr_hete_trans = nn.ModuleList()
        self.lr_hete_trans.append(nn.Linear((self.prop_steps + 1) * self.feat_dim, self.hidden_dim))

        for _ in range(num_layers - 2):
            self.lr_hete_trans.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lr_hete_trans.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            if self.bn:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.ln:
                self.norms.append(nn.LayerNorm(hidden_dim))

        self.softmax = nn.Softmax(dim=1)
        self.prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lr_hete_tran in self.lr_hete_trans:
            nn.init.xavier_uniform_(lr_hete_tran.weight, gain=gain)
            nn.init.zeros_(lr_hete_tran.bias)

    def forward(self, feature, learnable_re):

        for i in range(self.num_layers - 1):
            feature = self.lr_hete_trans[i](feature)
            if self.bn is True or self.ln is True:
                feature = self.norms[i](feature)
            feature = self.prelu(feature)
            feature = self.dropout(feature)
        feature_emb = self.lr_hete_trans[-1](feature)
        feature_emb_re = getre_scale(feature_emb)
        learnable_re = self.beta * learnable_re + (1 - self.beta) * feature_emb_re
        # learnable_re =  learnable_re
        learnable_re_mean = torch.mean(learnable_re)
        learnable_re_max = torch.max(learnable_re)

        learnable_re_pos_min = 0
        eps = torch.finfo(learnable_re.dtype).eps
        learnable_re_pos_difference = (learnable_re_max - learnable_re_mean - learnable_re_pos_min).clamp_min(eps)
        learnable_re_neg_min = -learnable_re_mean
        learnable_re_neg_difference = (0 - learnable_re_neg_min).clamp_min(eps)

        learnable_re = learnable_re - learnable_re_mean
        learnable_re = torch.where(learnable_re > 0,(learnable_re - learnable_re_pos_min) / learnable_re_pos_difference,
                                   -((learnable_re - learnable_re_neg_min) / learnable_re_neg_difference))

        learnable_re = add_diag(learnable_re)

        pos_signal = self.prelu(learnable_re)
        neg_signal = self.prelu(-learnable_re)


        prop_pos = self.softmax(torch.mm(pos_signal, feature_emb))
        prop_neg = self.softmax(torch.mm(neg_signal, feature_emb))

        local_message_propagation = (prop_pos - prop_neg) + feature_emb

        return local_message_propagation



class HetePropagateModel(nn.Module):
    def __init__(self, num_layers, feat_dim, hidden_dim, output_dim, prop_steps, dropout=0.5, bn=False, ln=False):
        super().__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.prop_steps = prop_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bn = bn
        self.ln = ln
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()

        self.lr_smooth_trans = nn.ModuleList()
        self.lr_smooth_trans.append(nn.Linear((self.prop_steps + 1) * self.feat_dim, self.hidden_dim))
        for _ in range(num_layers - 2):
            self.lr_smooth_trans.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lr_smooth_trans.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.lr_local_trans = nn.ModuleList()
        self.lr_local_trans.append(nn.Linear(self.feat_dim, self.hidden_dim))
        for _ in range(num_layers - 2):
            self.lr_local_trans.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lr_local_trans.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.hete_propagation = HetePropagateLayer(self.feat_dim, self.output_dim, self.prop_steps, self.hidden_dim, self.num_layers)

        self.norms = nn.ModuleList()
        if self.bn:
            for _ in range(num_layers - 1):
                self.norms.append(nn.BatchNorm1d(self.hidden_dim))
        if self.ln:
            for _ in range(num_layers - 1):
                self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lr_local_tran in self.lr_local_trans:
            nn.init.xavier_uniform_(lr_local_tran.weight, gain=gain)
            nn.init.zeros_(lr_local_tran.bias)
        for lr_smooth_tran in self.lr_smooth_trans:
            nn.init.xavier_uniform_(lr_smooth_tran.weight, gain=gain)
            nn.init.zeros_(lr_smooth_tran.bias)

    def forward(self, ori_feature, smoothed_feature, universal_re, device, is_train=False):
        ori_feature = ori_feature.to(device)
        smoothed_feature = smoothed_feature.to(device)
        universal_re = universal_re.to(device)
        input_prop_feature = smoothed_feature
        for i in range(self.num_layers - 1):
            smoothed_feature = self.lr_smooth_trans[i](smoothed_feature)
            if self.bn or self.ln:
                smoothed_feature = self.norms[i](smoothed_feature)
            smoothed_feature = F.relu(smoothed_feature)
            smoothed_feature = F.dropout(smoothed_feature, p=self.dropout.p, training=is_train)
        local_smooth_emb = self.lr_smooth_trans[-1](smoothed_feature)

        for i in range(self.num_layers - 1):
            ori_feature = self.lr_local_trans[i](ori_feature)
            if self.bn or self.ln:
                ori_feature = self.norms[i](ori_feature)
            ori_feature = F.relu(ori_feature)
            ori_feature = F.dropout(ori_feature, p=self.dropout.p, training=is_train)
        local_ori_emb = self.lr_local_trans[-1](ori_feature)

        local_message_propagation = self.hete_propagation(input_prop_feature, universal_re)


        emb_smooth = F.normalize(local_smooth_emb, p=2, dim=1)
        emb_ori = F.normalize(local_ori_emb, p=2, dim=1)
        emb_pro = F.normalize(local_message_propagation, p=2, dim=1)

        ano1 = mahalanobis_distance_normalized(emb_ori)
        ano2 = mahalanobis_distance_normalized(emb_smooth)
        ano3 = mahalanobis_distance_normalized(emb_pro)

        anomaly_score = (ano1 + ano2 + ano3) / 3

        return local_ori_emb, local_smooth_emb, local_message_propagation, anomaly_score


def getre_scale(emb):
    emb = F.normalize(emb, dim=1)
    re = emb @ emb.T
    re.fill_diagonal_(0)
    return re





def add_diag(re_matrix):
    re_matrix = re_matrix - torch.diag_embed(torch.diag(re_matrix))
    identity = torch.eye(
        re_matrix.shape[0],
        dtype=re_matrix.dtype,
        device=re_matrix.device,
    )
    return re_matrix + identity