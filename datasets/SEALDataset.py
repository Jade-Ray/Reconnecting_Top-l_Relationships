# Reconnecting Top-l Relationships 
# Copyright (c) 2022 JadeRay. All Rights Reserved.
# Licensed under the Creative Commons BY-NC-ND 4.0 International License [see LICENSE for details]
# --------------------------------------------------------------------------------------------------
# Modified from SEAL_OGB (https://github.com/facebookresearch/SEAL_OGB)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------------------------------------------------


from typing import List
import os.path as osp
from itertools import chain

import numpy as np
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path
import networkx as nx
import torch
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix

from utils import read_temporary_graph_data


class SEALDatasetInMemory(InMemoryDataset):
    """The SEAL formated Temporal Networks Dataset of `Stanford Large Network` in Memory.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (`"MathOverflow"`, 
            `"StackOverflow"`, `"WikiTalk"`, `"AskUbuntu"`, `"EmailEuCore"`).
        split (string, optional): The type of dataset split(`"train"`, 
            `"val"`, `"test"`, `"pred"`). (default: `"train"`)
        num_hop (int, optional): Number of hop subgraph around node. 
            (default: `2`)
        T (int, optional): Number of snapshots. (default: `100`)
    """
    
    url = 'https://snap.stanford.edu/data'
    info = {
        'EmailEuCore': {
            'file': "email-Eu-core-temporal.txt.gz",
            'timespan': 803,},
        'CollegeMsg': {
            'file': "CollegeMsg.txt.gz",
            'timespan': 193,},
        'MathOverflow': {
            'file': "sx-mathoverflow-a2q.txt.gz",
            'timespan': 2350,},
        'StackOverflow': {
            'file': "sx-stackoverflow-a2q.txt.gz",
            'timespan': 2774,},
        'WikiTalk': {
            'file': "wiki-talk-temporal.txt.gz",
            'timespan': 2320,},
        'AskUbuntu': {
            'file': "sx-askubuntu-a2q.txt.gz",
            'timespan': 2613,},}
    
    def __init__(self, root: str, name: str, split: str = "train", 
                 num_hops: int = 2, T: int = 100):
        self.name = name
        self.num_hop = num_hops
        self.T = T
        
        super().__init__(root)
        index = ['train', 'val', 'test'].index(split)
        self.data, self.slices = torch.load(self.processed_paths[index])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.info[self.name]["file"]]
    
    @property
    def raw_file_timespan(self) -> int:
        return self.info[self.name]["timespan"]
    
    @property
    def processed_file_names(self) -> str:
        return [f'SEAL_T{self.T}_train_data.pt', f'SEAL_T{self.T}_val_data.pt', 
                f'SEAL_T{self.T}_test_data.pt']
    @property
    def graph_data_file_name(self) -> str:
        return f'{self.raw_dir}/{self.raw_file_names[0]}'

    def download(self):
        download_url(f'{self.url}/{self.raw_file_names[0]}', self.raw_dir)

    def process(self):
        snapshots = read_temporary_graph_data(self.graph_data_file_name, self.raw_file_timespan, self.T)
        # filter num_nodes less than 100
        snapshots = list(filter(lambda s: s.number_of_nodes()>100, snapshots))  
        snapshots = list(map(from_networkx, snapshots))
        
        transform = RandomLinkSplit(num_val=0.05, num_test=0.1, split_labels=True)
        
        train_pos_data_list, train_neg_data_list= [], []
        val_pos_data_list, val_neg_data_list= [], []
        test_pos_data_list, test_neg_data_list= [], []
        self._max_z = 0
        
        # Collect a list of subgraphs for training, validation and testing:
        for snapshot in tqdm(snapshots, desc='Embedding'):
            train_data, val_data, test_data = transform(snapshot)
            train_pos_data_list += self.extract_enclosing_subgraphs(
                train_data.edge_index, train_data.pos_edge_label_index, train_data.num_nodes, 1)
            train_neg_data_list += self.extract_enclosing_subgraphs(
                train_data.edge_index, train_data.neg_edge_label_index, train_data.num_nodes, 0)
            
            val_pos_data_list += self.extract_enclosing_subgraphs(
                val_data.edge_index, val_data.pos_edge_label_index, val_data.num_nodes, 1)
            val_neg_data_list += self.extract_enclosing_subgraphs(
                val_data.edge_index, val_data.neg_edge_label_index, val_data.num_nodes, 0)
        
            test_pos_data_list += self.extract_enclosing_subgraphs(
                test_data.edge_index, test_data.pos_edge_label_index, test_data.num_nodes, 1)
            test_neg_data_list += self.extract_enclosing_subgraphs(
                test_data.edge_index, test_data.neg_edge_label_index, test_data.num_nodes, 0)
        
        # Convert node labeling to one-hot features.
        for data in chain(train_pos_data_list, train_neg_data_list,
                          val_pos_data_list, val_neg_data_list,
                          test_pos_data_list, test_neg_data_list):
            # We solely learn links from structure, dropping any node features:
            data.x = F.one_hot(data.z, self._max_z + 1).to(torch.float)
            
        torch.save(self.collate(train_pos_data_list + train_neg_data_list),
                   self.processed_paths[0])
        torch.save(self.collate(val_pos_data_list + val_neg_data_list),
                   self.processed_paths[1])
        torch.save(self.collate(test_pos_data_list + test_neg_data_list),
                   self.processed_paths[2])

    def extract_enclosing_subgraphs(self, edge_index, edge_label_index, num_nodes, y):
        data_list = []
        for src, dst in edge_label_index.t().tolist():
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hop, edge_index, relabel_nodes=True, num_nodes=num_nodes)
            src, dst = mapping.tolist()
            
            # Remove target link from the subgraph
            mask = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            sub_edge_index = sub_edge_index[:, mask]
            
            # Calculate node labeling.
            z = self.drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))
            
            data = Data(z=z, edge_index=sub_edge_index, y=y)
            data_list.append(data)
        
        return data_list
    
    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
        
        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]
        
        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]
        
        dist2src = shortest_path(adj_wo_dst, directed=True, unweighted=True, indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)
        
        dist2dst = shortest_path(adj_wo_src, directed=True, unweighted=True, indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)
        
        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2
        
        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1
        z[dst] = 1
        z[torch.isnan(z)] = 0.
        
        self._max_z = max(int(z.max()), self._max_z)
        
        return z.to(torch.long)
 
  
class SEALDataset(InMemoryDataset):
    """The SEAL formated Temporal Networks Dataset of `Stanford Large Network`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (`"MathOverflow"`, 
            `"StackOverflow"`, `"WikiTalk"`, `"AskUbuntu"`, `"EmailEuCore"`).
        split (string, optional): The type of dataset split(`"train"`, 
            `"val"`, `"test"`). (default: `"train"`)
        num_hop (int, optional): Number of hop subgraph around node. 
            (default: `2`)
        T (int, optional): Number of snapshots. (default: `100`)
    """
    
    url = 'https://snap.stanford.edu/data'
    info = {
        'MathOverflow': {
            'file': "sx-mathoverflow-a2q.txt.gz",
            'timespan': 2350,},
        'StackOverflow': {
            'file': "sx-stackoverflow-a2q.txt.gz",
            'timespan': 2774,},
        'WikiTalk': {
            'file': "wiki-talk-temporal.txt.gz",
            'timespan': 2320,},
        'AskUbuntu': {
            'file': "sx-askubuntu-a2q.txt.gz",
            'timespan': 2613,},}
    
    def __init__(self, root: str, name: str, split='train', pred_idx: int = -1, 
                 num_hops: int = 2, T: int = 100, max_z :int = None):
        self.name = name
        self.num_hop = num_hops
        self.T = T
        self.pred_idx = pred_idx
        self._max_z = 0 if max_z is None else max_z
        
        super().__init__(root)
        index = ['train', 'val', 'test'].index(split)
        self.data, self.slices = torch.load(self.processed_paths[index])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.info[self.name]["file"]]
    
    @property
    def raw_file_timespan(self) -> int:
        return self.info[self.name]["timespan"]
    
    @property
    def processed_file_names(self) -> str:
        return [f'SEAL_T{self.T}_Id{self.pred_idx}_train_data.pt', f'SEAL_T{self.T}_Id{self.pred_idx}_val_data.pt', 
                f'SEAL_T{self.T}_Id{self.pred_idx}_test_data.pt']

    def download(self):
        download_url(f'{self.url}/{self.raw_file_names[0]}', self.raw_dir)

    def process(self):
        snapshots = read_temporary_graph_data(f'{self.raw_dir}/{self.raw_file_names[0]}', self.raw_file_timespan, self.T)
        # filter num_nodes less than 100
        snapshots = list(filter(lambda s: s.number_of_nodes()>100, snapshots))
        snapshot = snapshots[self.pred_idx]
        snapshot = from_networkx(snapshot)
        
        transform = RandomLinkSplit(num_val=0.05, num_test=0.1, split_labels=True, neg_sampling_ratio=0.5)
        
        train_data, val_data, test_data = transform(snapshot)
        
        train_pos_data_list = self.extract_enclosing_subgraphs(
            train_data.edge_index, train_data.pos_edge_label_index, train_data.num_nodes, 1)
        train_neg_data_list = self.extract_enclosing_subgraphs(
            train_data.edge_index, train_data.neg_edge_label_index, train_data.num_nodes, 0)
            
        val_pos_data_list = self.extract_enclosing_subgraphs(
            val_data.edge_index, val_data.pos_edge_label_index, val_data.num_nodes, 1)
        val_neg_data_list = self.extract_enclosing_subgraphs(
            val_data.edge_index, val_data.neg_edge_label_index, val_data.num_nodes, 0)
        
        test_pos_data_list = self.extract_enclosing_subgraphs(
            test_data.edge_index, test_data.pos_edge_label_index, test_data.num_nodes, 1)
        test_neg_data_list = self.extract_enclosing_subgraphs(
            test_data.edge_index, test_data.neg_edge_label_index, test_data.num_nodes, 0)
        
        # Convert node labeling to one-hot features.
        for data in chain(train_pos_data_list, train_neg_data_list,
                          val_pos_data_list, val_neg_data_list,
                          test_pos_data_list, test_neg_data_list):
            # We solely learn links from structure, dropping any node features:
            data.x = F.one_hot(data.z, self._max_z + 1).to(torch.float)
            
        torch.save(self.collate(train_pos_data_list + train_neg_data_list),
                   self.processed_paths[0])
        torch.save(self.collate(val_pos_data_list + val_neg_data_list),
                   self.processed_paths[1])
        torch.save(self.collate(test_pos_data_list + test_neg_data_list),
                   self.processed_paths[2])

    def extract_enclosing_subgraphs(self, edge_index, edge_label_index, num_nodes, y):
        data_list = []
        for src, dst in edge_label_index.t().tolist():
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hop, edge_index, relabel_nodes=True, num_nodes=num_nodes)
            src, dst = mapping.tolist()
            
            # Remove target link from the subgraph
            mask = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            sub_edge_index = sub_edge_index[:, mask]
            
            # Calculate node labeling.
            z = self.drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))
            
            data = Data(z=z, edge_index=sub_edge_index, y=y)
            data_list.append(data)
        
        return data_list
    
    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
        
        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]
        
        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]
        
        dist2src = shortest_path(adj_wo_dst, directed=True, unweighted=True, indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)
        
        dist2dst = shortest_path(adj_wo_src, directed=True, unweighted=True, indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)
        
        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2
        
        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1
        z[dst] = 1
        z[torch.isnan(z)] = 0.
        
        self._max_z = max(int(z.max()), self._max_z)
        
        return z.to(torch.long)


def toSEAL_pred_datalist(graph: nx.Graph, num_features: int, num_hops: int = 2, neg_sampling_ratio: float = 1):
    
    def extract_enclosing_subgraphs(edge_index, edge_label_index, num_nodes, y, ori_nodes):
        data_list = []
        for src, dst in edge_label_index.t().tolist():
            pred_edge=torch.tensor([[ori_nodes[src], ori_nodes[dst]]], dtype=torch.long) # [1, 2] shape
            
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes)
            src, dst = mapping.tolist()
            
            # Remove target link from the subgraph
            mask = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            sub_edge_index = sub_edge_index[:, mask]
            
            # Calculate node labeling.
            z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))
                   
            data = Data(z=z, edge_index=sub_edge_index, y=y, pred_edge=pred_edge)
            data_list.append(data)
        
        return data_list
    
    def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
        
        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]
        
        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]
        
        dist2src = shortest_path(adj_wo_dst, directed=True, unweighted=True, indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)
        
        dist2dst = shortest_path(adj_wo_src, directed=True, unweighted=True, indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)
        
        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2
        
        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1
        z[dst] = 1
        z[torch.isnan(z)] = 0.
        
        return z.to(torch.long)
    
    original_nodes_list = list(graph.nodes)
    graph = from_networkx(graph)
    
    transform = RandomLinkSplit(num_val=0, num_test=0, split_labels=True, neg_sampling_ratio=neg_sampling_ratio)
    pred_data, _, _ = transform(graph)
    
    pred_pos_data_list = extract_enclosing_subgraphs(pred_data.edge_index, pred_data.pos_edge_label_index, pred_data.num_nodes, 1, original_nodes_list)
    pred_neg_data_list = extract_enclosing_subgraphs(pred_data.edge_index, pred_data.neg_edge_label_index, pred_data.num_nodes, 0, original_nodes_list)
    
    for data in chain(pred_pos_data_list, pred_neg_data_list):
        data.x = F.one_hot(data.z, num_features).to(torch.float)
    
    return pred_pos_data_list + pred_neg_data_list
