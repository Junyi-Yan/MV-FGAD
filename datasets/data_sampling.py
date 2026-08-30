import random
import warnings

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.convert import to_networkx

from datasets.structure_iid import structure_iid_louvain
from datasets.utils import idx_to_mask


def data_partitioning(G, sampling, num_clients, ratio_train, ratio_val, ratio_test):
    """Partition a node-level graph into client subgraphs with Louvain."""
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero")
    if not np.isclose(ratio_train + ratio_val + ratio_test, 1.0):
        raise ValueError("ratio_train, ratio_val, and ratio_test must sum to 1")
    if sampling != "Louvain":
        raise ValueError(f"Unsupported sampling method: {sampling}. Only Louvain is available.")

    graph_nx = to_networkx(G, to_undirected=True)
    node_dict = structure_iid_louvain(graph=graph_nx, num_clients=num_clients)

    return construct_subgraph_dict_from_node_dict(
        G=G,
        num_clients=num_clients,
        node_dict=node_dict,
        graph_nx=graph_nx,
        ratio_train=ratio_train,
        ratio_val=ratio_val)


def construct_subgraph_dict_from_node_dict(G,
    num_clients,
    node_dict,
    graph_nx,
    ratio_train,
    ratio_val):
    """Construct client subgraphs from a global-to-client node assignment."""

    subgraph_list = []
    client_nodes = {}
    client_splits = {}

    for client_id in range(num_clients):
        local_nodes = list(node_dict[client_id])
        client_nodes[client_id] = local_nodes.copy()
        num_local_nodes = len(local_nodes)
        local_idx = list(range(num_local_nodes))
        random.shuffle(local_idx)
        train_size = int(num_local_nodes * ratio_train)
        val_size = int(num_local_nodes * ratio_val)
        train_idx = local_idx[:train_size]
        val_idx = local_idx[train_size:train_size + val_size]
        test_idx = local_idx[train_size + val_size:]

        client_splits[client_id] = {"train": [local_nodes[i] for i in train_idx],
            "val": [local_nodes[i] for i in val_idx],
            "test": [local_nodes[i] for i in test_idx]}

    _ensure_anomaly_per_split(G=G,
        num_clients=num_clients,
        client_nodes=client_nodes,
        client_splits=client_splits)

    for client_id in range(num_clients):
        local_nodes = client_nodes[client_id]
        num_local_nodes = len(local_nodes)
        node_idx_map = {node_id: local_id for local_id, node_id in enumerate(local_nodes)}

        edges = []
        for source, target in graph_nx.subgraph(local_nodes).edges:
            edges.append((node_idx_map[source], node_idx_map[target]))
            edges.append((node_idx_map[target], node_idx_map[source]))

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_index, _ = add_self_loops(edge_index, num_nodes=num_local_nodes)

        train_mask = idx_to_mask([node_idx_map[node] for node in client_splits[client_id]["train"]],size=num_local_nodes)
        val_mask = idx_to_mask([node_idx_map[node] for node in client_splits[client_id]["val"]],size=num_local_nodes)
        test_mask = idx_to_mask([node_idx_map[node] for node in client_splits[client_id]["test"]],size=num_local_nodes)
        global_train = idx_to_mask(client_splits[client_id]["train"],size=G.y.size(0))
        global_val = idx_to_mask(client_splits[client_id]["val"],size=G.y.size(0))
        global_test = idx_to_mask(client_splits[client_id]["test"],size=G.y.size(0))

        subgraph = Data(x=G.x[local_nodes],
            y=G.y[local_nodes],
            edge_index=edge_index)
        subgraph.global_n_id = torch.tensor(local_nodes, dtype=torch.long)

        edge_array = edge_index.cpu().numpy()
        adjacency = sp.coo_matrix((np.ones(edge_index.size(1), dtype=np.float32),
                (edge_array[0], edge_array[1]),),shape=(num_local_nodes, num_local_nodes))
        subgraph.row = adjacency.row
        subgraph.col = adjacency.col
        subgraph.edge_weight = adjacency.data
        subgraph.adj = csr_matrix((subgraph.edge_weight,(subgraph.row, subgraph.col),),shape=(num_local_nodes, num_local_nodes))

        subgraph.train_idx = train_mask
        subgraph.val_idx = val_mask
        subgraph.test_idx = test_mask
        subgraph.global_train_idx = global_train
        subgraph.global_val_idx = global_val
        subgraph.global_test_idx = global_test

        train_normal, train_abnormal = count_labels(train_mask, subgraph.y)
        val_normal, val_abnormal = count_labels(val_mask, subgraph.y)
        test_normal, test_abnormal = count_labels(test_mask, subgraph.y)

        print(f"Client {client_id}: nodes={num_local_nodes}, edges={edge_index.size(1)}")
        print(f"  Train - normal: {train_normal}, abnormal: {train_abnormal}")
        print(f"  Val   - normal: {val_normal}, abnormal: {val_abnormal}")
        print(f"  Test  - normal: {test_normal}, abnormal: {test_abnormal}")

        subgraph_list.append(subgraph)

    return subgraph_list


def _ensure_anomaly_per_split(G,
    num_clients,
    client_nodes,
    client_splits):
    """Move anomalies between clients so each evaluable split has both classes."""
    for split_name in ("train", "val", "test"):
        for client_id in range(num_clients):
            split_nodes = client_splits[client_id][split_name]
            if any(G.y[node].item() != 0 for node in split_nodes):
                continue

            donor_id = None
            donor_anomalies = []

            for other_id in range(num_clients):
                if other_id == client_id:
                    continue

                candidates = [node
                    for node in client_splits[other_id][split_name]
                    if G.y[node].item() != 0]
                if len(candidates) > 1:
                    donor_id = other_id
                    donor_anomalies = candidates
                    break

            if donor_id is None:
                warnings.warn(f"Cannot place an anomaly in client {client_id} "
                    f"{split_name} split without emptying another client split.")
                continue

            chosen_node = random.choice(donor_anomalies)
            client_splits[client_id][split_name].append(chosen_node)
            client_nodes[client_id].append(chosen_node)
            client_splits[donor_id][split_name].remove(chosen_node)
            client_nodes[donor_id].remove(chosen_node)


def downsample(ds_rate, ds_cl, graphs):
    """Downsample one class in a graph-level dataset."""
    if ds_rate is None or ds_cl is None:
        return graphs
    if not 0 <= ds_rate <= 1:
        raise ValueError("ds_rate must be in [0, 1]")

    normal_graphs = [graph for graph in graphs if graph.y.item() != ds_cl]
    abnormal_graphs = [graph for graph in graphs if graph.y.item() == ds_cl]

    if not abnormal_graphs:
        return normal_graphs

    num_to_keep = max(1, int(len(abnormal_graphs) * ds_rate))
    sampled_abnormal_graphs = random.sample(abnormal_graphs, num_to_keep)
    return normal_graphs + sampled_abnormal_graphs


def _randChunk(
    graphs,
    num_client,
    overlap,
    ds_cl=None,
    seed=None,
    ds_rate=None):
    """Partition a graph-level dataset among clients."""

    if num_client <= 0:
        raise ValueError("num_client must be greater than zero")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    total_num = len(graphs)
    min_size = min(50, total_num // num_client)
    graph_chunks = []

    if not overlap:
        for client_id in range(num_client):
            start = client_id * min_size
            end = (client_id + 1) * min_size
            graph_chunks.append(list(graphs[start:end]))

        for graph in graphs[num_client * min_size:]:
            client_id = np.random.randint(low=0, high=num_client)
            graph_chunks[client_id].append(graph)
    else:
        if total_num == 0:
            return [[] for _ in range(num_client)]

        sizes = np.random.randint(low=50, high=150, size=num_client)
        graph_chunks = [random.choices(graphs, k=size) for size in sizes]

    return [downsample(ds_rate, ds_cl, graph_chunk) for graph_chunk in graph_chunks]


def count_labels(idx_mask, labels):
    """Count normal and abnormal labels selected by an index or mask."""
    indices = torch.as_tensor(idx_mask, device=labels.device)
    if indices.dtype == torch.bool:
        indices = torch.where(indices)[0]
    else:
        indices = indices.long()

    selected_labels = labels[indices]
    num_normal = (selected_labels == 0).sum().item()
    num_abnormal = (selected_labels == 1).sum().item()
    return num_normal, num_abnormal