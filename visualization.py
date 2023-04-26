from os.path import isfile
import pandas as pd
import numpy as np
import json
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import tensor
import random
from tools import check_para

global_index = 1


def save_figure(filename='seq', save_fig=False):
    global global_index
    path = './data/fig/figure_{}_{}.png'.format(filename, global_index)

    if save_fig:
        plt.savefig(path)
        plt.close('all')
        global_index += 1
    else:
        plt.show()


def show_graph(G, data=[], save_fig=False, name='seq'):
    data = np.array(data)
    node_num = G.number_of_nodes()
    # ndoe color
    node_label = G.ndata['L']
    # label_list

    label_num = node_label.size(1)
    label_list = [[] for i in range(label_num)]
    for lb in range(label_num):
        label_list[lb] = [i[lb] for i in node_label]

    # label_index
    label_index = [[] for i in range(label_num)]
    for index in range(label_num):
        label_index[index] = [i for i, x in enumerate(label_list[index]) if x == 1]

    plt.figure()
    img_G = G.to_networkx()

    if not data.any():
        pos = nx.kamada_kawai_layout(img_G)
    else:
        pos = {}
        for i in range(node_num):
            pos[i] = data[i]

    nx.draw(img_G, with_labels=True, node_size=100, pos=pos, arrows=False, width=0.5, font_size=8)

    # set color
    cmap = 'rbg'

    for i in range(label_num):
        nx.draw_networkx_nodes(img_G, pos, nodelist=label_index[i], node_color=cmap[i], node_size=100)
    #nx.draw_networkx_nodes(img_G, pos, nodelist=blue_index, node_color='b', node_size=100)  #, with_labels=True)
    #nx.draw_networkx_nodes(img_G, pos, nodelist=green_index, node_color='g', node_size=100)

    save_figure(filename=name, save_fig=save_fig)


def show_normal_graph(G, save_fig=False, name='seq'):
    global global_index

    #G = dgl.DGLGraph()
    img_G = G.to_networkx()
    pos = nx.kamada_kawai_layout(img_G)
    #pos = nx.circular_layout(img_G)
    nx.draw(img_G, with_labels=True, node_size=100, pos=pos, arrows=False, width=0.5, font_size=8)

    # edge weight
    #edge_weights = G.edata['w'].numpy()
    #print(edge_weights)
    #nx.draw_networkx_edges(img_G, pos, width=edge_weights * 2, arrows=False)

    save_figure(filename=name, save_fig=save_fig)


def show_test_data_graph(data, G, save_fig=False, name='seq'):
    node_num = G.number_of_nodes()

    #G = dgl.DGLGraph()
    img_G = G.to_networkx()

    # set node location on graph
    pos = {}
    for i in range(node_num):
        pos[i] = data[i]

    #pos = nx.circular_layout(img_G)
    nx.draw(img_G, with_labels=True, node_size=100, pos=pos, arrows=False, width=0.5, font_size=8)

    # edge weight
    edge_weights = G.edata['w'].numpy()
    #print(edge_weights)
    nx.draw_networkx_edges(img_G, pos, width=0.5, arrows=False)

    save_figure(filename=name, save_fig=save_fig)


def plot_acn_data(G, save_fig=False, name='seq'):

    save_figure(filename=name, save_fig=save_fig)


def show_raw_data(x, y, save_fig=False, name='seq'):
    plt.scatter(x, y)
    save_figure(filename=name, save_fig=save_fig)


def plot_ori_timeline(data, save_fig):
    x = data[0]
    y1 = data[1]
    y2 = data[2]

    fig = plt.figure(figsize=(10, 8))

    xtick = np.arange(1, len(data[0]), 15)
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, label='Connect Time')
    ax1.plot(x, y2, label='Charging Time')

    plt.xticks(xtick, rotation=50)

    plt.legend()

    save_figure('ori_time', save_fig=save_fig)


def plot_timeline(data, save_fig):
    y1 = data[2]
    y2 = data[3]
    y3 = data[4]
    y4 = data[5]
    y5 = data[6]
    x = data[1]

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(x, y1, '--g', label='Sessions Number')
    l2 = ax1.plot(x, y2, '--b', label='kWh Delivered')
    l3 = ax1.plot(x, y5, '--r', label='User Number')

    ax2 = ax1.twinx()
    ax2.set_ylim(np.min(np.array([y3, y4])), np.max(np.array([y3, y4])))
    l4 = ax2.plot(x, y3, label='Connect Time')
    l5 = ax2.plot(x, y4, label='Charging Time')

    ls = l1 + l2 + l3 + l4 + l5
    labs = [line.get_label() for line in ls]
    ax1.legend(ls, labs, loc=0)

    save_figure('new_time', save_fig=save_fig)
