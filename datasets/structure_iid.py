import math
import torch
from sklearn.cluster import spectral_clustering
from datasets.louvain import community_louvain


def _validate_partition_inputs(num_nodes, num_clients):
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero")
    if num_nodes <= 0:
        raise ValueError("The graph must contain at least one node")
    if num_clients > num_nodes:
        raise ValueError("num_clients cannot exceed the number of nodes")


def _split_oversized_communities(communities, max_size):
    chunks = []
    for nodes in communities:
        chunks.extend(nodes[start:start + max_size] for start in range(0, len(nodes), max_size))
    return chunks


def _assign_chunks_greedily(chunks, num_clients):
    owner_node_ids = {client_id: [] for client_id in range(num_clients)}
    client_sizes = [0] * num_clients

    for chunk in sorted(chunks, key=len, reverse=True):
        client_id = min(range(num_clients), key=lambda index: client_sizes[index])
        owner_node_ids[client_id].extend(chunk)
        client_sizes[client_id] += len(chunk)

    return owner_node_ids


def structure_iid_louvain(graph, num_clients, seed=None):
    num_nodes = graph.number_of_nodes()
    _validate_partition_inputs(num_nodes, num_clients)

    partition = community_louvain.best_partition(graph, random_state=seed)
    communities = {}
    for node_id, community_id in partition.items():
        communities.setdefault(community_id, []).append(node_id)

    max_chunk_size = math.ceil(num_nodes / num_clients)
    chunks = _split_oversized_communities(communities.values(), max_chunk_size)
    node_dict = _assign_chunks_greedily(chunks, num_clients)

    assigned_nodes = [node for nodes in node_dict.values() for node in nodes]
    if len(assigned_nodes) != num_nodes or len(set(assigned_nodes)) != num_nodes:
        raise RuntimeError("Louvain partition must assign every node exactly once")

    return node_dict


def structure_iid_sc(num_nodes, features, num_clients, seed=None):
    _validate_partition_inputs(num_nodes, num_clients)
    if features.shape[0] != num_nodes:
        raise ValueError("features.shape[0] must equal num_nodes")

    similarity = torch.sigmoid(features @ features.T).detach().cpu().numpy()
    cluster_labels = spectral_clustering(affinity=similarity,
        n_clusters=num_clients,
        random_state=seed)

    node_dict = {client_id: [] for client_id in range(num_clients)}
    for node_id, client_id in enumerate(cluster_labels.tolist()):
        node_dict[client_id].append(node_id)

    return node_dict