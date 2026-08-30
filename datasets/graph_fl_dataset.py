import os
import os.path as osp

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import TUDataset

from config import args
from datasets.data_sampling import _randChunk, data_partitioning
from datasets.utils import *

class GraphFLDataset(Dataset):
    def __init__(self, root, name, sampling, num_clients, analysis_local_subgraph, analysis_global_graph, ratio_train=0.2, ratio_val=0.4, ratio_test=0.4, transform=None, pre_transform=None, pre_filter=None, level="node"):
        self.root = os.path.join(root, name)
        self.name = name
        self.sampling = sampling
        self.num_clients = num_clients
        self.ratio_train = ratio_train
        self.ratio_val = ratio_val
        self.ratio_test = ratio_test
        self.analysis_local_subgraph = analysis_local_subgraph
        self.analysis_global_graph = analysis_global_graph
        self.level = level
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.load_data()

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.raw_dir, f"Client{self.num_clients}", self.sampling)

    @property
    def raw_file_names(self):
        return [f"{self.name}.mat"]

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    @property
    def processed_file_names(self):
        return [f"{self.name}_{i}.pt" for i in range(self.num_clients)]

    def get(self, idx):
        return torch.load(osp.join(self.processed_dir, f"{self.name}_{idx}.pt"))

    def load_global_graph(self):
        print(f"| ★ Load Global Data: {self.name}")
        if self.level == "node":
            mat_path = osp.join(self.raw_dir, f"{self.name}.mat")
            mat = sio.loadmat(mat_path)

            features = mat["Attributes"]
            if sp.issparse(features):
                features = features.toarray()
            features = torch.FloatTensor(features)
            labels = torch.as_tensor(mat["Label"].squeeze(), dtype=torch.long)
            adj = mat["Network"]
            if not sp.issparse(adj):
                adj = sp.coo_matrix(adj)
            row, col = adj.nonzero()
            edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)

            self.global_data = Data(x=features, y=labels, edge_index=edge_index)
            self.input_dim = features.shape[1]
            self.output_dim = args.output_dim
            self.labels = labels
        elif self.level == "graph":
            self.global_data = TUDataset(self.root, self.name)
            self.input_dim = self.global_data.num_features
            self.output_dim = self.global_data.num_classes
            self.labels = torch.tensor([g.y.item() for g in self.global_data])

    def process(self):
        self.load_global_graph()
        os.makedirs(self.processed_dir, exist_ok=True)

        if self.level == "node":
            subgraph_list = data_partitioning(G=self.global_data,
                sampling=self.sampling,
                num_clients=self.num_clients,
                ratio_train=self.ratio_train,
                ratio_val=self.ratio_val,
                ratio_test=self.ratio_test)

            for i in range(self.num_clients):
                subgraph = subgraph_list[i]

                if hasattr(subgraph, "global_n_id"):
                    subgraph.global_n_id = torch.as_tensor(subgraph.global_n_id, dtype=torch.long).clone()
                else:
                    subgraph.global_n_id = torch.arange(subgraph.num_nodes, dtype=torch.long)

                idx = subgraph.global_n_id
                subgraph.x = self.global_data.x[idx].clone()
                subgraph.y = self.global_data.y[idx].clone()
                torch.save(subgraph, self.processed_paths[i])
        elif self.level == "graph":
            raw_dataset = TUDataset(self.root, self.name)
            graphs = raw_dataset.shuffle()
            client_graphs = _randChunk(graphs, self.num_clients, overlap=False, ds_cl=1, ds_rate=0.2)

            for i, graphs in enumerate(client_graphs):
                torch.save(graphs, self.processed_paths[i])

    def load_data(self):
        if self.level == "node":
            self.load_global_graph()
            self.subgraphs = [self.get(i) for i in range(self.num_clients)]

            for i, subgraph in enumerate(self.subgraphs):
                if not hasattr(subgraph, "global_n_id"):
                    raise RuntimeError(f"Loaded subgraph {i} has no global_n_id")
                idx = torch.as_tensor(subgraph.global_n_id, dtype=torch.long)

                subgraph.x = self.global_data.x[idx].clone()
                subgraph.y = self.global_data.y[idx].clone()

                train_normal, train_abnormal = count_labels(subgraph.train_idx, subgraph.y)
                val_normal, val_abnormal = count_labels(subgraph.val_idx, subgraph.y)
                test_normal, test_abnormal = count_labels(subgraph.test_idx, subgraph.y)

                print(f"Client {i}:")
                print(f"  Train - normal: {train_normal}, abnormal: {train_abnormal}")
                print(f"  Val   - normal: {val_normal}, abnormal: {val_abnormal}")
                print(f"  Test  - normal: {test_normal}, abnormal: {test_abnormal}")

            self.global_data.train_idx = self.subgraphs[0].global_train_idx.clone()
            self.global_data.val_idx = self.subgraphs[0].global_val_idx.clone()
            self.global_data.test_idx = self.subgraphs[0].global_test_idx.clone()
            for subgraph in self.subgraphs[1:]:
                self.global_data.train_idx |= subgraph.global_train_idx
                self.global_data.val_idx |= subgraph.global_val_idx
                self.global_data.test_idx |= subgraph.global_test_idx

            if self.analysis_local_subgraph:
                for subgraph in self.subgraphs:
                    analysis_graph_structure_statis_info(subgraph)
                    analysis_graph_structure_homo_hete_info(subgraph)
        elif self.level == "graph":
            self.subgraphs = [self.get(i) for i in range(self.num_clients)]
            for i, graphs in enumerate(self.subgraphs):
                print(f"Client {i} - num graphs: {len(graphs)}")
                num_normal = sum(g.y.item() == 0 for g in graphs)
                num_abnormal = sum(g.y.item() != 0 for g in graphs)
                print(f"  normal: {num_normal}, abnormal: {num_abnormal}")

def count_labels(idx_mask, labels):
    idx = torch.as_tensor(idx_mask, device=labels.device)
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]
    else:
        idx = idx.long()
    selected_labels = labels[idx]
    num_normal = (selected_labels == 0).sum().item()
    num_abnormal = (selected_labels == 1).sum().item()
    return num_normal, num_abnormal