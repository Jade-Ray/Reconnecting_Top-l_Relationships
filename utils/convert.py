# Reconnecting Top-l Relationships 
# Copyright (c) 2022 JadeRay. All Rights Reserved.
# Licensed under the Creative Commons BY-NC-ND 4.0 International License [see LICENSE for details]

from pathlib import Path

import networkx as nx
import pandas as pd

import torch


def read_temporary_graph_data(filename, timespan: int, T: float):
    """Read temporary graph data from file.

    Args:
        filename (str or Path): the temporary graph data file path.
        timespan (int): the time span of data in days.
        T (float): the number of temporary graphs segmented with timestamp.

    Returns:
        List[nx.DiGraph]: the list of temporary direted graphs.
    """
    
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.exists():
        raise ValueError(f"{filename} does not exist.")

    if filename.suffix == '.gz':
        data = pd.read_csv(filename, compression='gzip', sep=' ', names=['SRC', 'TGT', 'TS'])
    else:
        raise ValueError(f"{filename.suffix} is not supported to read.")
    
    start_timestamp = int(data['TS'].min())
    interval = timespan * 24 * 3600 / T
    
    graphs = []
    
    for _ in range(T):
        end_timestamp = round(start_timestamp + interval)
        data_temp = data[data['TS'].between(start_timestamp, end_timestamp - 1)]
        graph = nx.from_pandas_edgelist(data_temp, 'SRC', 'TGT', create_using=nx.DiGraph)
        
        # remove isolate node and self loop
        graph.remove_nodes_from(nx.isolates(graph))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        
        graphs.append(graph)
        start_timestamp = end_timestamp
    
    return graphs


def read_graph_from_edgefile(filename):
    """Read files of edge index to DiGraph object.

    Args:
        filename (str or Path): the edge index file path.

    Returns:
        [nx.DiGraph]: the DiGraph object.
    """
    
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.exists():
        raise ValueError(f"{filename} does not exist.")
    
    pred_edge_index = torch.load(filename)
    pred_edge_index = pred_edge_index.numpy()
    DG = nx.DiGraph()
    DG.add_edges_from(pred_edge_index)
    return DG
