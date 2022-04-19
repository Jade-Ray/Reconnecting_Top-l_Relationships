# Reconnecting Top-l Relationships 
# Copyright (c) 2022 JadeRay. All Rights Reserved.
# Licensed under the Creative Commons BY-NC-ND 4.0 International License [see LICENSE for details]

import json
import math
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

def draw_networkx(graph: nx.Graph, iterations=15, seed=42, node_size=10, with_labels=False, arrows=False, width=0.15):
    """Draw networkx graph with spring layout.

    Args:
        graph (nx.Graph): the graph to draw.
        iterations (int, optional): max number of iterations to spring. Defaults to 15.
        seed (int, optional): random seed. Defaults to 42.
        node_size (int, optional): node size of draw. Defaults to 10.
        with_labels (bool, optional): whether draw labels. Defaults to False.
        arrows (bool, optional): whether draw arrows. Defaults to False.
        width (float, optional): edge width of draw. Defaults to 0.15.
    """
    
    pos = nx.spring_layout(graph, iterations=iterations, seed=seed)
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off')
    nx.draw_networkx(graph, pos=pos, ax=ax, arrows=arrows, 
                     node_size=node_size, with_labels=with_labels, width=width)
    
def draw_evaluation(record_path: str, type: str = 'elapsed', datasets=['MathOverflow', 'StackOverflow', 'WikiTalk', 'AskUbuntu'], algor=['SBG', 'CE-SBG', 'O_SBG'], mark=['ro-', 'bs-', 'gd-'], hatch =['.', 'x', '|'], save_path=None, figsize=(10, 8), suptitle=None, ncols=2):
    with open(record_path, 'r') as f:
        record = json.load(f)
    
    vary = record['vary']
    vary_list = record[vary]
    
    ncols = 1 if len(datasets) == 1 else ncols 
    nrows = math.ceil(len(datasets) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    if suptitle:
        fig.suptitle(suptitle)
    for i, dataset in enumerate(datasets):
        if len(datasets) == 1:
            ax = axs 
        elif len(datasets) == 2:
            ax = axs[i]
        else: 
            ax = axs[int(i/ncols), i%ncols]
        if type == 'elapsed':
            draw_efficiency_evaluation(ax, vary_list, record, dataset, algor, mark)
            ylabel = 'Time (s)'
        elif type == 'spread':
            draw_quality_evaluation(ax, vary_list, record, dataset, algor, hatch)
            ylabel = 'Influenced number'
        elif type == 'prob_edge': 
            draw_prob_edge(ax, vary_list, record, dataset, algor, hatch)
            ylabel = 'Number of Probing edges'
        
        # ax.set_title(dataset, y=1.04, fontsize='x-large')
        ax.set_title(dataset, y=1.08, fontsize='x-large')
        ax.set_ylabel(ylabel, fontsize='large')
        ax.set_xlabel(vary, fontsize='large')
        ax.legend(bbox_to_anchor=(0, 0.98, 1, 0.10), loc='lower left', ncol=3, mode='expand', edgecolor='1.0')
        ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True, pad=8, labelsize='large')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def draw_efficiency_evaluation(ax, vary_list, record_data, dataset, algor=['SBG', 'CE-SBG', 'O_SBG'], mark=['r+-', 'bs-', 'gd-']):
    for alg, m in zip(algor, mark):
        value = np.array(record_data[f'{dataset}_{alg}_elapsed'])
        
        value = np.where(value > 3600 * 24, 3600 * 24, value)
        valid_value = np.ma.masked_where(value >= 3600 * 24, value)
        invalid_value = np.ma.masked_where(value < 3600 * 24, value)
        
        ax.plot(vary_list, valid_value, m, label=alg, markersize=9, clip_on=False)
        # plot invalid value with *
        ax.plot(vary_list, invalid_value, f'{m[0]}*-', markersize=9, clip_on=False)
    
    ax.set_yscale('log', base=10, subs=[10])
    ax.set_yticks([10 ** i for i in range(-1, 6)])
    ax.set_ylim((0.1, 10 ** 5))
    ax.set_xticks(vary_list)
    ax.set_xlim((vary_list[0], vary_list[-1]))
        

def draw_quality_evaluation(ax, vary_list, record_data, dataset, algor=['SBG', 'CE-SBG', 'O_SBG'], hatch =['.', 'x', '|'], color=['#069AF3', '#F97306', '#15B01A']):
    x = np.arange(len(vary_list))
    width = 0.2
    for alg, h, c, label, offset in zip(algor, hatch, color, ['SBG', 'CE-SBG', 'O-SBG'], [-width, 0, width]):
        value = np.array(record_data[f'{dataset}_{alg}_spread'])
        value = np.where(value < 0, 1, value)
        
        ax.bar(x + offset, value, width, hatch=h, label=label)
    
    ax.set_yticks(np.arange(0, 30 * 11, 30))
    ax.set_ylim((0, 300))    
    ax.set_xticks(x, vary_list)


def draw_prob_edge(ax, vary_list, record_data, dataset, algor=['SBG', 'CE-SBG', 'O_SBG'], hatch =['.', 'x', '|'], color=['#069AF3', '#F97306', '#15B01A']):
    x = np.arange(len(vary_list))
    width = 0.2
    for alg, h, c, label, offset in zip(algor, hatch, color, ['SBG', 'CE-SBG', 'O-SBG'], [-width, 0, width]):
        value = np.array(record_data[f'{dataset}_{alg}_edges'])
        ax.bar(x + offset, value, width, hatch=h, label=label)
    
    ax.set_yscale('log', base=10, subs=[10])
    ax.set_yticks([10 ** i for i in range(0, 7)])
    ax.set_ylim((1, 10 ** 6))    
    ax.set_xticks(x, vary_list)