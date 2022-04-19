# Reconnecting Top-l Relationships 
# Copyright (c) 2022 JadeRay. All Rights Reserved.
# Licensed under the Creative Commons BY-NC-ND 4.0 International License [see LICENSE for details]

import time
import copy
import random
import heapq
from typing import List
from functools import reduce
import numpy as np
from tqdm import tqdm

import networkx as nx


def sketch_based_greedy_RTlL(graph: nx.Graph, l: int, users: List[int], ce: List[tuple[int, int]], R=200, snapshots=None, reduce_ce=False):
    """Algorithm 1: sketch-based greeedy (SBG) RTlL

    Args:
        graph (nx.Graph): the predicted snapshot graph Gt.
        l (int): the number of selected edges.
        users (List[int]): a group of user nodes.
        ce (List[tuple[int, int]]): a set of candidate edges.
        R (int, optional): the number of sketch subgraph generated. Defaults to 200.
        snapshots (List[nx.Graph], optional): the list of defined sketch subgraph. Defaults to None.
        reduce_ce (bool, optional): whether to reduce candidate edges. Defaults to False.

    Returns:
        solution, spreads, elapsed (list[edge], list[int], list[float]): the optimal reconnecting edge set, the spread number of these edge and the elapsed time.
    """
    
    elapsed, spreads, solution = [], [], []
    num_prob_edges = len(ce) if l==1 else l*len(ce) - reduce(lambda x, y: x + y, range(1, l))
    start_time = time.time()
    
    if not snapshots:
        # Generate R sketch subgraph Gjt where j ∈ [1,R]
        snapshots = generate_snapshots(graph, R)
    else:
        snapshots = copy.deepcopy(snapshots)
    
    pred_graph = graph.copy()
    candidate_edges = ce
    
    # find l edges with largest marginal gain
    pbar = tqdm(total=num_prob_edges, desc='Finding edge...')
    for _ in range(l):
        best_edge = (-1, -1)
        best_spread = -np.inf
        
        if reduce_ce:
            candidate_edges, reduce_num = reduce_CE(pred_graph, ce, users)
            num_prob_edges -= reduce_num
            pbar.update(reduce_num)
        
        # loop over edges that are not yet in our final solution to find biggest marginal gain
        edges = set(candidate_edges) - set(solution)  
        for edge in edges:
            spread = forward_influence_sketch(snapshots, users, edge)
            pbar.update()
            if spread > best_spread:
                best_spread, best_edge = spread, edge

        if best_edge == (-1, -1):
            print('There are no enough candidate edges to choose!')
            break
        
        solution.append(best_edge)
        spreads.append(best_spread)
        
        # add optimal edge in snapshot
        for snapshot in snapshots:
            unfrozen_graph = nx.DiGraph(snapshot)
            unfrozen_graph.add_edge(*best_edge)
            snapshot = unfrozen_graph
        # update predict graph
        pred_graph.add_edge(*best_edge)
        
        elapsed.append(round(time.time() - start_time, 3))
    
    pbar.close()
    return solution, spreads, elapsed, num_prob_edges


def reduce_CE(graph: nx.Graph, ce: List[tuple[int, int]], users: List[int]):
    """Reducing candidate edge that not related users

    Args:
        graph (nx.Graph): the predicted snapshot graph Gt.
        ce (List[tuple[int, int]]): a set of candidate edges.
        users (List[int]): a group of user nodes.

    Returns:
        reduced_ce, reduce_num (List[tuple[int, int]], int): a set of reduced candidate edges and reduce num.
    """

    _, reached_node = compute_independent_cascade(graph, users)
            
    reduced_ce = [e for e in ce if e[0] in reached_node or e[1] in reached_node]
    return reduced_ce, len(ce) - len(reduced_ce)


def order_based_SBG_RTlL(graph: nx.Graph, l: int, users: List[int], ce: List[tuple[int, int]], UBL: tuple[dict, list[nx.Graph]]):
    """Algorithm 4: Order-based SBG RTlL

    Args:
        graph (nx.Graph): graph object
        l (int): the number of reconneted edges
        users (List[int]): a group of user nodes
        ce (List[tuple[int, int]]): a set of candidate edges
        UBL (tuple[dict, list[nx.Graph]]): the UBL index (L, Gsg).
        
    Returns:
        solution, spreads, elapsed, lookups (list[edge], list[int], list[float]): the optimal reconnecting edge set, the spread number of these edge, the elapsed time and the number of lookup in search.
    """
    
    start_time = time.time()
    
    solution, spreads, lookups, elapsed = [], [], [], []
    UB1, snapshots = UBL
    predict_graph = graph.copy()
    candidate_edges, _ = reduce_CE(predict_graph, ce, users) # reducing CE #BFS
    unwork_ce = set(ce) - set(candidate_edges)
    
    # initial second-setp upper bound
    gains = []
    for e in candidate_edges:
        UB2_upper_bound, _ = UB1[e]
        heapq.heappush(gains, (-UB2_upper_bound, e))
    
    # generate a mask to mark the nodes which are reached by users
    mask = []
    for snapshot in snapshots:
        mask.append(compute_independent_cascade(snapshot, users)[1])
    
    for _ in tqdm(range(l), desc='Finding edge...'):
        edge_lookup = 0
        matched = False
        
        if len(gains) == 0:
            print('There are no enough candidate edges to choose!')
            elapsed.append(round(time.time() - start_time, 3))
            break
        
        while not matched:
            # count the number of times the spread is computed
            edge_lookup += 1
            # recalculate spread of top node
            _, current_edge = heapq.heappop(gains)
            # evaluate the spread function and store the marginal gain
            spread_gain = sketch_estimate(snapshots, current_edge, mask, UB1)
            
            # check if the previous top edge stayed on the top after pushing
            heapq.heappush(gains, (-spread_gain, current_edge))
            matched = gains[0][1] == current_edge
        
        # spread stores the cumulative spread
        spread_gain, edge = heapq.heappop(gains)
        
        # update new candidate edges with optimal edge
        predict_graph.add_edge(*edge)
        new_ce, _ = reduce_CE(predict_graph, unwork_ce, users)
        for e in new_ce:
            UB2_upper_bound, _ = UB1[e]
            heapq.heappush(gains, (-UB2_upper_bound, e))
        unwork_ce -= set(new_ce)
        
        update_mask(snapshots, edge, mask)
       
        solution.append(edge)
        spreads.append(-spread_gain)
        lookups.append(edge_lookup)
        elapsed.append(round(time.time() - start_time, 3))
        
    return solution, spreads, elapsed, lookups


def forward_influence_sketch(graphs: List[nx.Graph], users: List[int], reconneted_edge: tuple[int, int]):
    """The Forward Influence Sketch method

    Args:
        graphs (List[nx.Graph]): a set of snapshot graph
        users (List[int]): a group of user nodes
        reconneted_edge (tuple[int, int]): the reconneted edge

    Returns:
        spread (float): the mean of additional spread of vertexes reached by users in all sketch subgraph.
    """
    
    spread = []
    for graph in graphs:
        original_spread = compute_independent_cascade(graph, users)[0]
        graph_temp = graph.copy()
        graph_temp.add_edge(*reconneted_edge)
        new_spread = compute_independent_cascade(graph_temp, users)[0]
        spread.append(new_spread - original_spread)
    
    return np.mean(spread)


def compute_independent_cascade(graph: nx.Graph, users: List[int], mask: List[int] = []):
    """Compute independent cascade in the graph

    Args:
        graph (nx.Graph): graph object
        users (List[int]): a set of user nodes
        mask (List[int], optional): a set of unparticipation cascade mask nodes. default is []. 

    Returns:
        spread, reached_node (int, List[int]): the number of vertexes reached by users in graph and the set of nodes reached by users.
    """
    
    new_active, active = users[:], users[:]
        
    # for each newly activated nodes, find its neighbors that becomes activated
    while new_active:
        activated_nodes = []
        for node in new_active:
            if graph.has_node(node):
                # determine neighbors that become infected
                neighbors = list(graph.neighbors(node))
                activated_nodes += neighbors
        
        # ensure the newly activated nodes doesn't already exist
        new_active = list(set(activated_nodes) - set(active) - set(mask))
        active += new_active

    return len(active), active


def generate_snapshots(graph: nx.Graph, r: int, seed: int = 42):
    """Generate r random sketch graph by removing each edges with probability 1-P(u,v), which defined as 1/degree(v). 

    Args:
        graph (nx.Graph): graph object
        r (int): the number of snapshots generated
        seed (int): the random seed of numpy

    Returns:
        snapshots (List[nx.Graph]): r number sketch subgraph
    """

    np.random.seed(seed)
    snapshots = []
    for _ in range(r):
        select_edges = [edge for edge in graph.edges if np.random.uniform(0, 1) < 1/graph.degree(edge[1])]
        snapshots.append(graph.edge_subgraph(select_edges))

    return snapshots


def build_upper_bound_label(graph: nx.Graph, ce: List[tuple[int, int]], R=200):
    """Bulid upper bound label (UBL) index for each candidate edge

    Args:
        graph (nx.Graph): the predicted snapshot graph Gt.
        ce (List[tuple[int, int]]): a set of candidate edge.
        R (int): the number of sketch subgraph generated. Defaults to 200.

    Returns:
        UBL, snapshots (dict[edge, (upper_bound, flag)], list[nx.Graph]): the upper bound label and r number sketch subgraph.
    """
    
    snapshots = generate_snapshots(graph, R)
    
    UBL = {}
    for u, v in ce:
        UBL[(u, v)] = (compute_independent_cascade(graph, [v])[0], False)
    
    return (UBL, snapshots)


def sketch_estimate(graphs: List[nx.Graph], reconneted_edge: tuple[int, int], mask: List[List[int]], UB1: dict):
    """The influence spread expansion estimation

    Args:
        graphs (List[nx.Graph]): a set of snapshot graph
        reconneted_edge (tuple[int, int]): the reconneted edge
        mask (List[List[int]]): a set of unparticipation cascade mask nodes
        UB1 (dict): the upper bound first

    Returns:
        spread (float): the mean of spread of vertexes reached by edge but fail to influenced by mask in all sketch subgraph.
    """
    
    _, UB1_flag = UB1[reconneted_edge]
    u, v = reconneted_edge
    spread, spread_record = 0, 0
    for i, graph in enumerate(graphs):
        graph_temp = graph.copy()
        graph_temp.add_edge(*reconneted_edge)
            
        # fetch the spread but fail to influenced by mask
        if u in mask[i] and v not in mask[i]:
            spread += compute_independent_cascade(graph_temp, [v], mask=mask[i])[0]
            
        # record edge of not flag spread to new UB1_upper_bound
        if not UB1_flag:
            spread_record += compute_independent_cascade(graph_temp, [v])[0]

    # Update Upper Bound First
    if not UB1_flag:
        UB1[(u, v)] = (spread_record / len(graphs), True)
    
    return spread / len(graphs)


def update_mask(graphs: List[nx.Graph], reconneted_edge: tuple[int, int], mask: List[List[int]]):
    """Update the mask of nodes reached by edge

    Args:
        graphs (List[nx.Graph]): a set of snapshot graph
        reconneted_edge (tuple[int, int]): the reconneted edge
        mask (List[List[int]]): a set of unparticipation cascade mask nodes
    """
    
    u, v = reconneted_edge
    for i, graph in enumerate(graphs):
        if graph.has_node(u) and u in mask[i] and v not in mask[i]:
            graph_temp = graph.copy()
            graph_temp.add_edge(*reconneted_edge)
            
            _, reached_node = compute_independent_cascade(graph_temp, [v], mask=mask[i])
            mask[i] += reached_node


def calculate_candidate_edges(pred_graph, total_graph):
    """Calculate candidate edges from pred graph and original graphs.

    Args:
        pred_graph (str): the predicted graph.
        total_graph (str): the original temporary graphs as one graph.

    Returns:
        set: The candidated edges.
    """
    
    total_edges = set(total_graph.edges)
    pred_edges = set(pred_graph.edges)
    
    candidate_edges = total_edges - (total_edges & pred_edges)
    predict_graph_nodes = list(pred_graph.nodes)
    candidate_edges = list(filter(lambda e: e[0] in predict_graph_nodes and e[1] in predict_graph_nodes, candidate_edges))
    
    return candidate_edges


def generate_user_groups(graph, group_size):
    return generate_user_groups_helper(graph, [], list(graph.nodes), group_size)


def generate_user_groups_helper(graph, user_group, candiate_nodes, group_size):
    """The recursive function to calculate user group with conneted users.

    Args:
        graph (nx.DiGraph): the DiGraph object.
        user_group (list): the list of users group.
        candiate_nodes (list): the candiate list of nodes.
        group_size (int): the number of user group

    Returns:
        List[int]: the random generated conneted users group. 
    """
    if len(user_group) >= group_size:
        return user_group
    random.shuffle(candiate_nodes)
    if not candiate_nodes:
        user_group = user_group[:-1]
        return user_group
    for node in candiate_nodes:
        if len(user_group) >= group_size:
            break
        if node not in user_group:
            user_group.append(node)
            user_group = generate_user_groups_helper(graph, user_group, list(graph.neighbors(node)), group_size)       
    return user_group
