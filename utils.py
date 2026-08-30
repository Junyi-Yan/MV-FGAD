import os
import os.path as osp
import random
from ctypes import c_int
from sklearn.metrics import roc_auc_score, precision_recall_curve,average_precision_score
import numpy as np
import numpy.ctypeslib as ctl
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

def sparseTensor_to_coomatrix(edge_idx, num_nodes):
    if edge_idx.numel() == 0:
        return sp.coo_matrix((num_nodes, num_nodes), dtype=np.int64)

    row = edge_idx[0].detach().cpu().numpy()
    col = edge_idx[1].detach().cpu().numpy()
    data = np.ones(edge_idx.shape[1], dtype=np.int64)
    return sp.coo_matrix(
        (data, (row, col)),
        shape=(num_nodes, num_nodes),
        dtype=np.int64)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse_coo_tensor(indices,values,size=sparse_mx.shape,dtype=torch.float32).coalesce()


def adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype)
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        left = np.power(degrees, r - 1)
        right = np.power(degrees, -r)
    left[~np.isfinite(left)] = 0.0
    right[~np.isfinite(right)] = 0.0

    return sp.diags(left) @ adj @ sp.diags(right)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        inverse_sqrt = np.power(degrees, -0.5)
    inverse_sqrt[~np.isfinite(inverse_sqrt)] = 0.0
    degree_matrix = sp.diags(inverse_sqrt)
    return (degree_matrix @ adj @ degree_matrix).tocoo()


def _load_matmul_library(library_name):
    module_dir = osp.dirname(osp.abspath(__file__))
    return ctl.load_library(osp.join("models", "csrc", library_name), module_dir)


def _prepare_csr_inputs(adj, feature):
    adj = adj.tocsr()
    feature = np.ascontiguousarray(feature, dtype=np.float32)
    data = np.ascontiguousarray(adj.data, dtype=np.float32)
    indices = np.ascontiguousarray(adj.indices, dtype=np.int32)
    indptr = np.ascontiguousarray(adj.indptr, dtype=np.int32)
    output = np.zeros(feature.size, dtype=np.float32)
    return output, data, indices, indptr, feature


def csr_sparse_dense_matmul(adj, feature):
    library = _load_matmul_library("libmatmul.so")
    int_array = ctl.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")
    float_array = ctl.ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS")
    library.FloatCSRMulDenseOMP.argtypes = [float_array,float_array,int_array,int_array,float_array,c_int,c_int]
    library.FloatCSRMulDenseOMP.restype = None
    output, data, indices, indptr, feature = _prepare_csr_inputs(adj, feature)
    rows, columns = feature.shape
    library.FloatCSRMulDenseOMP(output,data,indices,indptr,feature.ravel(),rows,columns)
    return output.reshape(feature.shape)


def cuda_csr_sparse_dense_matmul(adj, feature):
    library = _load_matmul_library("libcudamatmul.so")
    int_array = ctl.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")
    float_array = ctl.ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS")
    library.FloatCSRMulDense.argtypes = [float_array,c_int,float_array,int_array,int_array,float_array,c_int,c_int]
    library.FloatCSRMulDense.restype = c_int

    output, data, indices, indptr, feature = _prepare_csr_inputs(adj, feature)
    rows, columns = feature.shape
    library.FloatCSRMulDense(output,len(data),data,indices,indptr,feature.ravel(),rows,columns)
    return output.reshape(feature.shape)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _min_max_normalize(values, eps):
    minimum = values.min()
    maximum = values.max()
    return (values - minimum) / (maximum - minimum + eps)


def cosine_similarity_anomaly_score(features, eps=1e-8):
    center = features.mean(dim=0, keepdim=True)
    similarity = F.cosine_similarity(features, center, dim=1, eps=eps)
    return _min_max_normalize(1.0 - similarity, eps)


def mahalanobis_distance_normalized(X, eps=1e-8):
    mu = X.mean(dim=0, keepdim=True)        # (1, D)
    X_centered = X - mu                     # (N, D)
    cov = (X_centered.T @ X_centered) / (X_centered.size(0) - 1)
    cov_inv = torch.linalg.pinv(cov)
    left = X_centered @ cov_inv
    md = torch.sqrt(torch.sum(left * X_centered, dim=1) + eps)
    md_min = md.min()
    md_max = md.max()
    md_norm = (md - md_min) / (md_max - md_min + eps)
    return md_norm


def euclidean_distance_normalized(features, eps=1e-8):
    center = features.mean(dim=0, keepdim=True)
    distance = torch.linalg.vector_norm(features - center, ord=2, dim=1)
    return _min_max_normalize(distance, eps)


def project_1d(features):
    projected = PCA(n_components=1).fit_transform(features.detach().cpu().numpy())
    return projected.squeeze()


def get_sparse_att(self, adj, wh):
    del self
    adj = adj.coalesce()
    indices = adj.indices()
    similarity = F.cosine_similarity(wh[indices[0]], wh[indices[1]])
    positive = F.relu(similarity)
    negative = -F.relu(-similarity)
    positive_adj = torch.sparse_coo_tensor(indices, positive, adj.size()).coalesce()
    negative_adj = torch.sparse_coo_tensor(indices, negative, adj.size()).coalesce()
    return positive_adj, negative_adj

def auc_ap(labels, output, positive_class=0):
    if isinstance(output, np.ndarray):
        output = torch.tensor(output)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    output = output.detach().cpu()
    labels = labels.detach().cpu().numpy()
    probs = output.numpy()
    labels = labels.astype(np.int64)
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    return auc, ap

def mahalanobis_score(emb, normal_mask, eps=1e-8):
    emb_n = emb[normal_mask]
    mu = emb_n.mean(dim=0, keepdim=True)
    Xc_n = emb_n - mu
    cov = (Xc_n.T @ Xc_n) / (Xc_n.size(0) - 1)
    cov_inv = torch.linalg.pinv(cov)
    Xc = emb - mu
    md = torch.sqrt(torch.sum((Xc @ cov_inv) * Xc, dim=1) + eps)
    return md


def build_train_class_masks(train_idx, labels, device):
    train_idx = torch.as_tensor(train_idx, device=device)
    if train_idx.dtype == torch.bool:
        if train_idx.numel() != labels.numel():
            raise ValueError("Boolean train_idx must have the same length as labels")
        train_mask = train_idx
    else:
        train_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=device)
        train_mask[train_idx.long()] = True

    normal_mask = train_mask & (labels == 0)
    abnormal_mask = train_mask & (labels != 0)

    if normal_mask.sum() < 2:
        raise ValueError("At least two normal training nodes are required")
    if not abnormal_mask.any():
        raise ValueError("At least one abnormal training node is required")

    return normal_mask, abnormal_mask


def single_consistency_loss(emb,normal_mask,abnormal_mask,margin=1.0,eps=1e-8):
    md = mahalanobis_score(emb, normal_mask, eps)
    md_n = md[normal_mask]
    md_a = md[abnormal_mask]
    loss_normal = md_n.mean()
    loss_abnormal = torch.relu(margin - (md_a.mean() - md_n.mean()))
    return loss_normal + loss_abnormal


def consistency_loss(local_emb,smooth_emb,hete_emb,normal_mask,abnormal_mask,eps=1e-8,margin=1.0):
    loss_local = single_consistency_loss(local_emb, normal_mask, abnormal_mask, margin, eps)
    loss_smooth = single_consistency_loss(smooth_emb, normal_mask, abnormal_mask, margin, eps)
    loss_hete = single_consistency_loss(hete_emb, normal_mask, abnormal_mask, margin, eps)
    return loss_local + loss_smooth + loss_hete

def train(model, train_idx, labels, device, optimizer):
    model.train()
    optimizer.zero_grad()
    emb, anomaly_score = model.model_forward(torch.arange(model.processed_feature.size(0)), device)
    normal_mask, abnormal_mask = build_train_class_masks(train_idx,labels,device)
    loss_train = single_consistency_loss(emb,normal_mask,abnormal_mask,margin=1.0,eps=1e-8)
    y_true = labels[train_idx].cpu().numpy()
    y_score = anomaly_score[train_idx].detach().cpu().numpy()
    auc_train, ap_train = auc_ap(y_true, y_score)
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), auc_train, anomaly_score

def evaluate(model, train_idx, val_idx, test_idx, labels, device):
    model.eval()
    with torch.no_grad():
        _, anomaly_score = model.model_forward(idx=torch.arange(labels.size(0), device=device),device=device)

        auc_val, ap_val = auc_ap(labels[val_idx].cpu().numpy(),anomaly_score[val_idx].cpu().numpy())
        auc_test, ap_test = auc_ap(labels[test_idx].cpu().numpy(),anomaly_score[test_idx].cpu().numpy())
    return auc_val, auc_test, ap_val, ap_test

class BaseTask:
    def __init__(self):
        pass

    def _execute(self):
        return NotImplementedError

    def _evaluate(self):
        return NotImplementedError

    def _train(self):
        return NotImplementedError

