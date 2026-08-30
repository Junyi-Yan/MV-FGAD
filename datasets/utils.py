import networkx as nx
import torch
import numpy as np
import os.path as osp
from scipy.spatial import distance
from torch_geometric.utils.convert import to_networkx
from collections import Counter


def analysis_graph_structure_statis_info(G):
    structure_statis_info = {}
    y = G.y
    structure_statis_info["num_nodes"] = G.num_nodes
    structure_statis_info["num_edges"] = G.num_edges
    structure_statis_info["average_degree"] = G.num_edges/G.num_nodes
    structure_statis_info["density"] = G.num_edges / (G.num_nodes * (G.num_nodes - 1))
    return structure_statis_info


def analysis_graph_structure_homo_hete_info(G):
    structure_homo_hete_label_info = {}
    structure_homo_hete_label_info["node_homophily"] = label_node_homogeneity(G)
    structure_homo_hete_label_info["edge_homophily"] = label_edge_homogeneity(G)
    structure_homo_hete_feature_info = {}
    return structure_homo_hete_label_info, structure_homo_hete_feature_info


def average_shortest_path_length_for_all(G):
    tmp_G=G.copy()
    isolate_nodes_num = len(list(nx.isolates(tmp_G)))
    if nx.is_connected(G):
        average = nx.average_shortest_path_length(tmp_G)
    else:
        iso_nodes = nx.isolates(G)
        tmp_G.remove_nodes_from(iso_nodes)
        if nx.is_connected(tmp_G):
            average = nx.average_shortest_path_length(tmp_G)
        else:
            subgraphs = list(tmp_G.subgraph(i) for i in list(nx.connected_components(tmp_G)))
            average = 0
            for sb in subgraphs:
                average += nx.average_shortest_path_length(sb)
            average /= (len(subgraphs)*1.0)
    return average, isolate_nodes_num

def label_node_homogeneity(G):
    num_nodes = G.num_nodes
    homophily = 0
    for edge_u in range(num_nodes):
        hit = 0
        edge_v_list = G.edge_index[1][torch.where(G.edge_index[0] == edge_u)]
        if len(edge_v_list) != 0:
            for i in range(len(edge_v_list)):
                edge_v = edge_v_list[i]
                if G.y[edge_u] == G.y[edge_v]:
                    hit += 1
            homophily += hit / len(edge_v_list)
    homophily /= num_nodes
    return homophily

def label_edge_homogeneity(G):
    num_edges = G.num_edges
    homophily = 0
    for i in range(num_edges):
        if G.y[G.edge_index[0][i]] == G.y[G.edge_index[1][i]]:
            homophily += 1
    homophily /= num_edges
    return homophily

def feature_node_homogeneity(G):
    num_nodes = G.num_nodes
    homophily = 0
    for edge_u in range(num_nodes):
        sim_list = []
        hit = 0
        edge_v_list = G.edge_index[1][torch.where(G.edge_index[0] == edge_u)]
        if len(edge_v_list) != 0:
            for i in range(len(edge_v_list)):
                edge_v = edge_v_list[i]
                sim = (1 - distance.cosine(G.x[edge_u], G.x[edge_v]))
                sim_list.append(sim)
                hit += sim
            hit /= len(edge_v_list)
            sim_min = min(sim_list)
            sim_max = max(sim_list)
            if (sim_max - sim_min) != 0:
                homophily += (hit-sim_min) / (sim_max - sim_min)
            else:
                homophily += hit
    homophily /= num_nodes
    return homophily

def feature_edge_homogeneity(G):
    num_edges = G.num_edges
    homophily = 0
    sim_list = []
    for i in range(num_edges):
        sim = (1 - distance.cosine(G.x[G.edge_index[0][i]], G.x[G.edge_index[1][i]]))
        sim_list.append(sim)
        homophily += sim
    homophily /= num_edges
    sim_list = list(filter((0.0).__ne__, sim_list))
    sim_list = list(filter((1).__ne__, sim_list))
    sim_min = min(sim_list)
    sim_max = max(sim_list)
    homophily = (homophily-sim_min) / (sim_max - sim_min)
    return homophily

def idx_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def remove_duplicate_two_dimension_list_element(input_list):
    tmp_list = []
    for two_dimension in input_list :
        if two_dimension not in tmp_list:
            tmp_list.append(two_dimension)
    return tmp_list
