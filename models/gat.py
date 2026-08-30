import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import sparse_mx_to_torch_sparse_tensor, adj_to_symmetric_norm


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)



class SpGraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[0]
        edge = adj.nonzero().t()
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout=0.5):
        super(GAT, self).__init__()
        self.use_graph_op = True
        self.pre_graph_op = None
        self.gat1 = SpGraphAttentionLayer(in_features=feat_dim, out_features=hidden_dim, dropout=dropout)
        self.gat2 = SpGraphAttentionLayer(in_features=hidden_dim, out_features=output_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.processed_feature = None
        self.adj = None
        self.post_graph_op = None

    def preprocess(self, adj, feature):
        self.pre_msg_learnable = False
        self.processed_feature = feature
        adj = adj_to_symmetric_norm(adj, r=0.5)
        self.adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()

    def forward(self, idx, device):
        x = self.processed_feature.to(device)
        adj = self.adj.to(device)
        x = self.gat1(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, adj)
        return x[idx]

    def model_forward(self, idx, device):
        return self.forward(idx, device)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)
        return output
