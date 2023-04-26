import os
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
from math import pi as PI
import math
from scipy import sparse

from visualization import *


def generate_k_degree_regular_graph(N, k):
    # degree should be eq or less than N-1
    if k > N - 1 and (N * k) % 2 == 1:
        return False

    # initial node buffer
    node_list = [[] for i in range(k + 1)]
    node_list[0] = [i for i in range(N)]
    # node_list[2].append(7)

    #print(node_list)

    # generate edges
    head_list = []
    tail_list = []

    def distribute_edge(k):
        #cur = 0
        head = []
        tail = []

        target_degree, target_node = [(k - 1 - idx, item[0]) for idx, item in enumerate(node_list[k - 1::-1]) if item][0]
        node_list[target_degree].remove(target_node)

        # fullfill target node
        for i in range(k - target_degree):
            head.append(target_node)

            tail_idx, tail_node = [(idx, item[0]) for idx, item in enumerate(node_list[:k]) if item][0]
            tail.append(tail_node)

            # reset head node and tail node to correct list
            node_list[tail_idx].remove(tail_node)
            node_list[tail_idx + 1].append(tail_node)

        # reset head node
        node_list[k].append(target_node)
        #print(node_list)
        return head, tail

    while len(node_list[k]) != N:
        head, tail = distribute_edge(k)

        head_list.extend(head)
        tail_list.extend(tail)

    #print(head_list, tail_list)

    return head_list, tail_list


def generate_line_graph(N):
    head_list, tail_list = generate_k_degree_regular_graph(N, 2)
    head_list.pop()
    tail_list.pop()
    return head_list, tail_list


# design two pair of graph connected by lines
def connected_graph(p1_node, p1_degree, p2_node, p2_degree, line_node):
    p1_node_list = [i for i in range(p1_node * 2)]
    p2_ndoe_list = [i + p1_node * 2 for i in range(p2_node * 2)]
    line_node_list = [i + p1_node * 2 + p2_node * 2 for i in range(line_node)]

    #print(p1_node_list, p2_ndoe_list, line_node_list)

    def list_add_number(list_a, num):
        return list(np.array(list_a) + num)

    p1_head_list, p1_tail_list = generate_k_degree_regular_graph(p1_node, p1_degree)
    p1_head_list.extend(list_add_number(p1_head_list, p1_node))
    p1_tail_list.extend(list_add_number(p1_tail_list, p1_node))

    p1_sum = p1_node * 2
    p2_head_list, p2_tail_list = generate_k_degree_regular_graph(p2_node, p2_degree)
    p2_head_list.extend(list_add_number(p2_head_list, p2_node))
    p2_tail_list.extend(list_add_number(p2_tail_list, p2_node))
    p2_head_list = list_add_number(p2_head_list, p1_sum)
    p2_tail_list = list_add_number(p2_tail_list, p1_sum)

    # connected line pick intersection point
    p1_p2_sum = p1_sum + p2_node * 2
    inter_point = line_node_list[0]

    # fix strategy
    line_head, line_tail = generate_line_graph(4)
    line_head = list_add_number(line_head, p1_p2_sum)
    line_tail = list_add_number(line_tail, p1_p2_sum)

    # add chunk
    line_head.extend([inter_point, inter_point])
    line_tail.extend(line_node_list[-2:])
    # connect whole graph
    connecter_head = line_node_list[-4:]
    connected_tail = [p1_node_list[0], p1_node_list[-1], p2_ndoe_list[0], p2_ndoe_list[-1]]

    head_list = p1_head_list + p2_head_list + line_head + connecter_head
    tail_list = p1_tail_list + p2_tail_list + line_tail + connected_tail
    node_num = len(p1_node_list + p2_ndoe_list + line_node_list)

    # dgl cuild graph
    G = dgl.DGLGraph()
    G.add_nodes(node_num)
    G.add_edges(head_list, tail_list)

    # set label
    labels = torch.zeros((node_num, 2))
    labels[p1_node_list[1]][1] = 1
    labels[p1_node_list[5]][1] = 1
    labels[p2_ndoe_list[1]][0] = 1
    labels[p2_ndoe_list[6]][0] = 1

    G.ndata['L'] = labels
    G.edata['w'] = torch.ones(len(head_list))
    G = dgl.add_reverse_edges(G, copy_edata=True)

    if False:
        show_graph(G)

    return G


def build_test_graph(node_num, image=True, degree=False):
    G = dgl.DGLGraph()

    if not degree:
        head, tail = generate_line_graph(node_num)
    else:
        head, tail = generate_k_degree_regular_graph(node_num, degree)

    G.add_nodes(node_num)
    G.add_edges(head, tail)
    #print(type(G))

    # weights initial
    # cut into two pieces, we want two nodes share different distribution
    clA_num = int(node_num / 2)
    class_A = torch.randn((clA_num, 10))
    class_B = torch.normal(2, 0.5, size=(node_num - clA_num, 10))
    x_fea = torch.cat((class_A, class_B), dim=0)
    G.ndata['fea'] = x_fea

    # random inital node label
    A_index = random.sample(range(0, clA_num), 1)
    B_index = random.sample(range(clA_num, node_num), 1)
    A_index.extend(B_index)
    node_index = A_index
    #print(node_index)

    # initial edges weight
    edge_weights = torch.rand(len(head))

    edge_weights = torch.cat((edge_weights, edge_weights), 0)

    G = dgl.add_reverse_edges(G)  # to_bidirected(G)
    G.edata['w'] = edge_weights
    #print(G.adj())
    G.nodes[node_index].data['L'] = torch.tensor([[1, 0], [0, 1]])

    if image:
        show_graph(G, save_fig=True)

    return G


def distance(data):
    node_num = len(data)
    dist_mat = np.zeros((node_num, node_num))

    for i in range(node_num):
        for j in range(node_num):
            dist_mat[i][j] = np.linalg.norm(data[i] - data[j])

    return dist_mat


def kNN(dist_mat, k=5, set_labels=False):
    print('use kNN build test graph')
    node_num = dist_mat.shape[0]

    zero_index = np.argwhere(dist_mat == 0)
    for loc in zero_index:
        if loc[0] != loc[1]:
            pass
    #print(np.nonzero(dist_mat)[0].shape)
    G = dgl.DGLGraph()

    # set edges
    adj_mat = np.zeros((node_num, node_num))
    edge_weights = []
    for source in range(node_num):
        targets = dist_mat[source].argsort()[:k + 1]
        for target in targets:
            adj_mat[source][target] = 1
            # edge_weights.append(dist_mat[source][target])

    # remove self loop
    for i in range(node_num):
        adj_mat[i][i] = 0

    # set bi-graph and set weight to 0
    adj_mat = adj_mat + adj_mat.T
    adj_mat = np.where(adj_mat > 0, 1, 0)

    head, tail = np.where(adj_mat == 1)

    G.add_edges(head, tail)
    for a, b in zip(head, tail):
        edge_weights.append(dist_mat[a][b])
    G.edata['w'] = torch.tensor(edge_weights)

    #print(G.edata['w'])
    #print(len(G.edata['w']))
    # set label
    if set_labels:
        labels = torch.zeros((node_num, 2))
        labels[9][0] = 1
        labels[10][0] = 1
        labels[15][0] = 1
        labels[-1][1] = 1
        labels[-3][1] = 1
    else:
        labels = torch.zeros((node_num, 2))

    G.ndata['L'] = labels

    #show_normal_graph(G, True)
    print('use kNN build test graph done')
    return G


def e_distance(dist_mat, thersh=1):
    print('use e-distance build test graph')
    node_num = dist_mat.shape[0]
    G = dgl.DGLGraph()

    head = []
    tail = []

    adj_mat = np.zeros((node_num, node_num))
    for source in range(node_num):
        for target in range(node_num):
            if source != target and dist_mat[source][target] < thersh:
                head.append(source)
                tail.append(target)

    G.add_nodes(node_num)
    G.add_edges(head, tail)
    G.edata['w'] = torch.ones(len(head))

    # set label
    labels = torch.zeros((node_num, 2))
    labels[0][0] = 1
    labels[-1][1] = 1
    G.ndata['L'] = labels

    #show_normal_graph(G, True)
    print('use e-distance build test graph done')
    return G


# generate test data set
def generate_test_set(node_num=200, ring_per=0.4, load=False):
    local_path = './data/fig/{}.csv'.format(node_num)
    if os.path.exists(local_path) and load:
        print('load test dataset')
        data = np.loadtxt(local_path, delimiter=',')
        labels = data[:, 0]
    else:
        print('generate test dataset')
        ring_range = (7, 8)
        cir_range = (0, 3)

        def generate_coor(r_range):
            r = random.uniform(r_range[0], r_range[1])
            agl = random.uniform(0, 2 * PI)

            return r * math.cos(agl), r * math.sin(agl)

        labels = []
        ring_x = []
        ring_y = []
        # generate ring coor
        ring_num = int(node_num * ring_per)
        for i in range(ring_num):
            x, y = generate_coor(ring_range)
            ring_x.append(x)
            ring_y.append(y)
            labels.append(0)

        cir_x = []
        cir_y = []
        # circle radius [0,2]
        cir_num_down = int((node_num - ring_num) / 2)
        for i in range(cir_num_down):
            x, y = generate_coor(cir_range)
            cir_x.append(x)
            cir_y.append(y - 4)
            labels.append(1)

        cir_num_up = int(node_num - ring_num - cir_num_down)
        for i in range(cir_num_up):
            x, y = generate_coor(cir_range)
            cir_x.append(x)
            cir_y.append(y + 4)
            labels.append(1)

        x_cor = np.array(ring_x + cir_x)
        y_cor = np.array(ring_y + cir_y)
        labels = np.array(labels)

        # visualize

        data = np.array([labels, x_cor, y_cor]).T
        # write to file
        np.set_printoptions(suppress=True)
        np.savetxt(local_path, data, delimiter=',')

    vis_data = data.T
    show_raw_data(vis_data[1], vis_data[0], True)
    print('test dataset done')
    return data, labels


def set_labels(G, input_labels=[], show_all=False, label_rate=0.05):
    input_labels = np.array(input_labels)
    node_num = G.number_of_nodes()

    if not input_labels.any():
        labels = torch.zeros((node_num, 2))
        labels[9][0] = 1
        labels[10][0] = 1
        labels[15][0] = 1
        labels[-1][1] = 1
        labels[-3][1] = 1
        print('no label')
    else:
        label_num = len(set(input_labels))
        labels = torch.zeros((node_num, label_num))
        # set labels
        if show_all:
            set_label_idx = [i for i in range(node_num)]
        else:
            set_label_idx = np.array([])
            for lb in range(label_num):
                tmp_labels = np.where(input_labels == lb)[0]
                pre_label_num = int(tmp_labels.shape[0] * label_rate)
                pre_label_num = 10
                ini_label = 0
                set_label_idx = np.append(set_label_idx, tmp_labels[ini_label:ini_label + pre_label_num])
                # set_label_idx = np.append(set_label_idx, np.array([53, 73]))
                #print(pre_label_num)
                #print(set_label_idx)

            # print(set_label_idx)

        #print(set_label_idx)
        for idx in set_label_idx:
            #print(input_labels[idx])
            labels[int(idx)][int(input_labels[int(idx)])] = 1

    G.ndata['L'] = labels
    #print(labels)

    return G


def real_graph(data, set_all=False):
    labels = np.array(data[:, 0])
    attributes = data[:, 1:]

    dist_mat = distance(attributes)

    G = kNN(dist_mat, k=11)
    #G = e_distance(dist_mat, thersh=con_thresh)

    G = set_labels(G, labels, set_all)
    #G = dgl.add_reverse_edges(G, copy_edata=True)

    return G
