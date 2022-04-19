# Reconnecting Top-l Relationships 
# Copyright (c) 2022 JadeRay. All Rights Reserved.
# Licensed under the Creative Commons BY-NC-ND 4.0 International License [see LICENSE for details]

"""
Reconnecting Top-l Relationships (RTlR) query
"""

import random
import json
import os.path as osp
from collections import defaultdict
from functools import reduce

import numpy as np

from utils import sketch_based_greedy_RTlL, order_based_SBG_RTlL, build_upper_bound_label, calculate_candidate_edges, generate_user_groups
from utils import read_temporary_graph_data, read_graph_from_edgefile
from utils import draw_networkx, draw_evaluation


var_dict = {
    'num_query': [20, 40, 60, 80, 100],
    'user_group_size': [1, 2, 4, 6, 8],
    'reconneted_edge_size': [1, 50, 100, 150],
    'snapshot_size': [20, 40, 60, 80, 100],
    }
default_values = {'num_query': 80, 'user_group_size': 6, 'reconneted_edge_size': 10, 'snapshot_size': 100}
datasets = {
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
        'timespan': 2613,},
    'EmailEuCore': {
        'file': "email-Eu-core-temporal.txt.gz",
        'timespan': 803,},
    'CollegeMsg': {
        'file': "CollegeMsg.txt.gz",
        'timespan': 193,},}

np.random.seed(42)
random.seed(42)


def efficiency_evaluation(predict_graph, candidate_edges, num_query=40, user_group_size=6, 
                          reconneted_edge_size=10, snapshot_size=100,
                          save_path=None, algs=['O_SBG', 'CE-SBG', 'SBG']):
    reconneted_edge_size = min(reconneted_edge_size, len(candidate_edges))
    all_elapsed = defaultdict(list)
    all_spreads = defaultdict(list)
    all_num_prob_edges = defaultdict(list)
    
    print(f'\nEvaluation...')
    for alg in algs:
        if save_path is not None and osp.exists(f"{save_path}_{alg}.json"):
            with open(f"{save_path}_{alg}.json", 'r') as f:
                temp = json.load(f)
                all_elapsed[alg] = temp['elapsed']
                all_spreads[alg] = temp['spreads']
                all_num_prob_edges[alg] = temp['num_prob_edges']
            continue
                
        if alg == 'O_SBG':
            UBL = build_upper_bound_label(predict_graph, candidate_edges)
                
        print(f'{alg} algorithm...')
        for i in range(num_query):
            print(f"[{i+1}/{num_query}] query processing...")
            user_group = generate_user_groups(predict_graph, user_group_size)
            if user_group_size == 1:
                while len(list(predict_graph.neighbors(user_group[0]))) < 2:
                    print(f'user group {user_group} has no neighbors.')
                    user_group = generate_user_groups(predict_graph, user_group_size)
            print(f'random user group are: {user_group}')
                
            if alg == 'O_SBG':
                _, spreads, elapsed, lookups = order_based_SBG_RTlL(predict_graph, l=reconneted_edge_size, users=user_group, ce=candidate_edges, UBL=UBL)
                num_prob_edges = sum(lookups)
            elif alg == 'CE-SBG':
                _, spreads, elapsed, num_prob_edges =  sketch_based_greedy_RTlL(predict_graph, l=reconneted_edge_size, users=user_group, ce=candidate_edges, reduce_ce=True)
            elif alg == 'SBG':
                _, spreads, elapsed, num_prob_edges = sketch_based_greedy_RTlL(predict_graph, l=reconneted_edge_size, users=user_group, ce=candidate_edges)
            
            all_elapsed[alg].append(elapsed[-1])
            all_spreads[alg].append(sum(spreads))
            all_num_prob_edges[alg].append(num_prob_edges)
            print(f'{alg}\telapsed: {elapsed[-1]} \tspreads: {sum(spreads)} \tnum_prob_edges: {num_prob_edges}')
        
        if save_path is not None and not osp.exists(f"{save_path}_{alg}.json"):
            with open(f"{save_path}_{alg}.json", "w") as outfile:
                json.dump({'elapsed': all_elapsed[alg], 'spreads': all_spreads[alg], 'num_prob_edges': all_num_prob_edges[alg]}, outfile)
    
    return all_elapsed, all_spreads, all_num_prob_edges


def generate_record_dict(vary: str, vary_list: list):
    assert vary in list(default_values.keys())
    
    record = defaultdict(list) 
    record['vary'] = vary
    for item in default_values.keys():
        if item == vary:
            record[item] = vary_list
        else:
            record[item] = default_values[item]
            
    return record


def load_dataset(name: str, value: dict, snapshot_size: int):  
    if name in ['StackOverflow', 'WikiTalk', 'CollegeMsg']:
        predict_graph = read_temporary_graph_data(f'data/SEALDataset/{name}/raw/{value["file"]}', value["timespan"], snapshot_size)[-1]
        total_time_span = value["timespan"] * (snapshot_size - 1) / snapshot_size
        total_graph = read_temporary_graph_data(f'data/SEALDataset/{name}/raw/{value["file"]}', total_time_span, 1)[0]
        algs=['O_SBG', 'CE-SBG', 'SBG']
    else:
        predict_graph = read_graph_from_edgefile(f'data/SEALDataset/{name}/T{snapshot_size}_pred_edge.pt')
        total_graph = read_temporary_graph_data(f'data/SEALDataset/{name}/raw/{value["file"]}', value["timespan"], 1)[0]
        algs=['O_SBG', 'CE-SBG', 'SBG']
        
    return predict_graph, total_graph, algs

def run(vary: str):
    if vary in var_dict.keys():
        vary_list = var_dict[vary]
    else:
        raise ValueError(f"Not support vary {vary}")
    
    for dataset, value in datasets.items():
        if osp.exists(f"data/varying_{vary}_{dataset}.json"):
            print(f"{dataset} experiment already done.\n")
            continue
        print(f'Experiment in {dataset} Dataset...')
        
        predict_graph, total_graph, algs = load_dataset(dataset, value, default_values['snapshot_size'])
        
        record = generate_record_dict(vary, vary_list)  
        for v in vary_list:
            print(f"\nVarying {vary} experiment with {v}")
            
            if vary == 'snapshot_size':
                predict_graph, total_graph, algs = load_dataset(dataset, value, v)
                
            candidate_edges = calculate_candidate_edges(predict_graph, total_graph)
            
            temp_save_path = f'data/temp/{dataset}/{dataset}_{vary}_{v}'
            elapsed, spreads, num_prob_edges = efficiency_evaluation(predict_graph, candidate_edges, **{vary: v, 'save_path': temp_save_path}, algs=algs)
            for algor in elapsed.keys():
                record[f'{dataset}_{algor}_elapsed'].append(np.mean(elapsed[algor]))
                record[f'{dataset}_{algor}_spread'].append(np.mean(spreads[algor]))
                record[f'{dataset}_{algor}_edges'].append(np.mean(num_prob_edges[algor]))

        with open(f"data/varying_{vary}_{dataset}.json", "w") as outfile:
            json.dump(record, outfile)


def vis_output():
    datasets = ['MathOverflow', 'AskUbuntu', 'CollegeMsg', 'EmailEuCore']
    type = ['elapsed', 'spread']
    json_files = ['data/varying_num_query.json', 'data/varying_reconneted_edge_size.json', 
                'data/varying_snapshot_size.json', 'data/varying_user_group_size.json']
    save_path = {
        'elapsed': ['result/varying_nq.png', 'result/varying_re.png', 
                    'result/varying_ss.png', 'result/varying_us.png',],
        'spread': ['result/quality_nq.png', 'result/quality_re.png', 
                'result/quality_ss.png', 'result/quality_us.png',]
    }
    for t in type:
        for jf, sp in zip(json_files, save_path[t]):
            draw_evaluation(jf, type=t, datasets=datasets, save_path=sp)
            if 'reconneted_edge_size' in jf:
                draw_evaluation(jf, type='prob_edge', datasets=datasets, save_path='result/num_prob_edges.png')

            
if __name__ == '__main__':
    run('reconneted_edge_size')
    run('snapshot_size')
    run('user_group_size')
    run('num_query')
    
    vis_output()
