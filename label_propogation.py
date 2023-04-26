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
from collections import Counter
from math import exp, log, sqrt
from math import pi as PI
import time

from tools import *
from GCN import *
from visualization import *

global_index = 1
inf = float('inf')


def build_laplacian_mat(G, weight=False):
    # node num
    node_num = G.number_of_nodes()
    # extract adjcency matrix
    sparse_adj_mat = G.adj()
    lap_mat = sparse_adj_mat.to_dense()

    # extract degree matrix
    degrees = G.in_degrees([i for i in range(node_num)])
    #print(degrees)

    # extract edge weight
    #print(G.all_edges())
    head, tail = G.all_edges()

    for (h, t) in zip(head, tail):
        if weight:
            lap_mat[h][t] = -G.edges[h, t].data['w']
        else:
            lap_mat[h][t] = -1

    for i in range(node_num):
        lap_mat[i][i] = degrees[i]

    lap_mat = lap_mat.numpy()
    eg_value, eg_vector = np.linalg.eig(lap_mat)

    # visualization
    plt.figure()
    img_G = G.to_networkx()
    pos = nx.kamada_kawai_layout(img_G)
    nx.draw(img_G, with_labels=True, node_size=100, pos=pos, arrows=False, width=0.5, font_size=8)
    plt.show()
    plt.close()

    # line graph
    plt.plot([i for i in range(node_num)], eg_vector[2])
    plt.show()

    return lap_mat


def calculate_modularity(G):
    # reconstruct adjacency
    A = G.adj().to_dense()
    m = G.num_edges()
    k = G.in_degrees()
    K = torch.outer(k, k) / (2 * m)
    labels = G.ndata['L']
    sig = torch.matmul(labels, labels.T)
    Q = 1 / (2 * m) * torch.sum((A - K) * sig.float())
    return Q.numpy()


# normal, GCN, s2v
def label_propagation(G, threshold=0.01, epoch_num=3, emb='GCN', reset=True, view_arg=False, data=[]):
    ori_node_num = G.number_of_nodes()
    # time start
    start_t = time.time()

    # extract adjcency matrix
    if emb == 'normal':
        sparse_adj_mat = G.adj()
        adj_mat = sparse_adj_mat.to_dense()
    elif emb == 'GCN':
        F = compute_filter(G)
        X = G.ndata['fea']
        emb_X = torch.mm(F, X)
        adj_mat = node_similarity(G, emb_X)
    elif emb == 's2v':
        # k, _ = Floyd(G)
        #print('k from Floyd: {}'.format(k))
        #check_para(k)
        k = 20

        strcu2_t0 = time.time()
        MG = struc2vec(G, k, view_arg=view_arg)
        struc2_t1 = time.time()
        print('struc2vec ', struc2_t1 - strcu2_t0)
        #check_para(MG)
        label = G.ndata['L']
        label_num = label.size()[1]

        ori_G = G
        ori_node_num = ori_G.number_of_nodes()

        if ori_node_num != MG.number_of_nodes():
            for i in range(k):
                label = torch.cat((label, G.ndata['L']), dim=0)

        MG.ndata['L'] = label
        G = MG
        adj_mat = G.adj().to_dense()
    else:
        return

    show_graph(G, data=data, save_fig=True, name='prss_{}nds'.format(ori_node_num))

    set_label_t0 = time.time()
    node_num = G.number_of_nodes()
    head, tail = G.all_edges()
    for (h, t) in zip(head, tail):
        # thinkg of add thresh hold here
        adj_mat[h][t] = G.edges[h, t].data['w']
    set_label_t1 = time.time()
    print('set label time ', set_label_t1 - set_label_t0)

    # transform matrix
    tm_t0 = time.time()
    ones = torch.ones(node_num)
    P = adj_mat / torch.matmul(adj_mat, ones)
    # P = torch.where(torch.isnan(P), torch.full_like(P, 0), P)
    tm_t1 = time.time()
    print('build transform matrix', tm_t1 - tm_t0)

    # label information initial
    F = G.ndata['L'].to(dtype=torch.float)
    loss = 10
    res = F
    epoch = 0
    labeled_index_num = 0
    label_num = F.size(1)

    # propagation process
    ini_index = torch.nonzero(F)
    # check_para(ini_index)

    lp_time0 = time.time()
    # while loss > threshold and epoch < epoch_num:
    while labeled_index_num < node_num and epoch < epoch_num:
        epoch += 1
        last_res = res
        res = torch.mm(P, last_res)

        #print(res[:10])
        #print(res.size())

        #print(res)
        # regularization on probability
        res = res.T
        reg_base = torch.sum(res, dim=0)  # res[0] + res[1] + res[2]
        # print(reg_base)
        res = res / reg_base
        res = res.T
        res = torch.where(torch.isnan(res), torch.full_like(res, 0), res)

        #print(res[:10])
        #print(torch.nonzero(res))

        #print(res)
        # claculate the loss
        loss = torch.norm(last_res.view(-1) - res.view(-1), p=2)
        # show loss
        print('loss: ', loss.numpy())

        # reset initial label, set probability of initial nodes to 1
        if reset:
            for labeled in ini_index:
                res.numpy()[labeled[0]] = F[labeled[0]]

        # attribute new label to garph
        #check_para(res)
        ones = torch.ones(label_num)
        labeled_index = torch.nonzero(torch.mv(res, ones)).reshape(-1).numpy()

        #print(torch.mv(res, ones))
        labeled_index_num = len(labeled_index)

        label_index = torch.argmax(res, 1)
        new_label = torch.zeros(node_num, label_num)

        # check_para(new_label)

        for i in range(node_num):
            if i in labeled_index:
                new_label[i][label_index[i]] = 1

        # check_para(label_index)

        G.ndata['L'] = new_label
        # res = new_label

        # ------------ fix all
        # ini_index = torch.nonzero(new_label)
        # F = new_label

        # heck_para(ini_index)
        # print(len(ini_index))
        # visualize
        show_graph(G, data=data, save_fig=True, name='prss_{}nds'.format(ori_node_num))
        lp_time1 = time.time()
        print('epoch {} take time: {}'.format(epoch, lp_time1 - lp_time0))
        lp_time0 = lp_time1

        # tranform matrix regularization

    # reconstruc origin garph
    if emb == 's2v':
        split_layer = res.split(ori_node_num, 0)
        re_cons_prob = sum(split_layer)
        #check_para(re_cons_prob)
        re_cons_index = torch.argmax(re_cons_prob, 1)
        #check_para(re_cons_index)
        re_label = torch.zeros(ori_node_num, label_num)
        for i in range(ori_node_num):
            re_label[i][re_cons_index[i]] = 1
        #check_para(re_label)
        #print(prob_mat)
        #label_mat = G.ndata['L'].reshape(k + 1, ori_node_num, 2)
        #print(G.ndata['L'].size())
        ori_G.ndata['L'] = re_label

        show_graph(ori_G, data=data, save_fig=True, name='s2v_{}_{}nds'.format(str(view_arg), ori_node_num))
        G = ori_G

    return G


# optimized DTW use a tuple to avoid too much repetitive
def DTW(x, y, opt=True):
    # compress sequences
    if opt:
        x_count = Counter(x)
        y_count = Counter(y)
        x = [(key, x_count[key]) for key in x_count]
        y = [(key, y_count[key]) for key in y_count]

    #check_para(x)
    #check_para(y)

    # build dist matrix
    origin_dist_mat = np.zeros((len(x), len(y)))
    #print(origin_dist_mat)
    for i in range(len(x)):
        for j in range(len(y)):
            if not opt:
                origin_dist_mat[i][j] = max(x[i], y[j]) / min(x[i], y[j]) - 1
            else:
                origin_dist_mat[i][j] = (max(x[i][0], y[j][0]) / min(x[i][0], y[j][0]) - 1) * max(x[i][1], y[j][1])

    #print(origin_dist_mat)

    # recursively get distance at (i,j)
    def recur_dist(m, n):
        #print(m, n)
        if m == 0 and n > 0:
            return recur_dist(m, n - 1) + origin_dist_mat[m][n]
        elif n == 0 and m > 0:
            return recur_dist(m - 1, n) + origin_dist_mat[m][n]
        elif m == 0 and n == 0:
            return origin_dist_mat[m][n]
        else:
            return min(recur_dist(m - 1, n) + origin_dist_mat[m][n], recur_dist(m, n - 1) + origin_dist_mat[m][n], recur_dist(m - 1, n - 1) + 2 * origin_dist_mat[m][n])

    res = recur_dist(len(x) - 1, len(y) - 1)
    return res


# DTW replacement
def pad_norm(x, y):
    x, y = x.copy(), y.copy()
    # pad short list with zero
    zeros = [0 for i in range(abs(len(x) - len(y)))]

    if len(x) > len(y):
        y.extend(zeros)
    else:
        x.extend(zeros)

    x, y = np.array(x), np.array(y)

    return np.linalg.norm(x - y, ord=2)


# Floyd Algorithm - get diameter of a graph
def Floyd(G):
    # reconstruct adjacency matrix
    #print(G.adj())
    adj_mat = G.adj().to_dense().numpy()
    node_num = G.number_of_nodes()
    inf = float('inf')

    # initial dist matrix
    dist_mat = np.where(adj_mat == 0, inf, adj_mat)
    # print(dist_mat)
    for i in range(node_num):
        dist_mat[i][i] = 0

    #print(adj_mat[-1])
    #print(dist_mat)
    t0 = time.time()
    # build floyd algorithm
    for k in range(node_num):
        tk0 = time.time()
        for i in range(node_num):
            ti0 = time.time()
            for j in range(node_num):
                dist_mat[i][j] = min(dist_mat[i][j], dist_mat[i][k] + dist_mat[k][j])
            ti1 = time.time()
            #print('i', ti1 - ti0)

        tk1 = time.time()
        #print('k: ', k, tk1 - tk0)

    # calculate mean
    flat_mat = np.triu(dist_mat).flatten()
    flat_mat = flat_mat[flat_mat != 0]
    flat_mat = flat_mat[flat_mat != inf]
    max_dist = int(flat_mat.max())
    #print(np.unique(flat_mat))
    #print(max_dist)
    #print(flat_mat)
    edge_sum = sum(flat_mat)
    edge_num = np.nonzero(flat_mat)[0].shape
    #print('avg route length ', (edge_sum / edge_num)[0], 'max_length ', max_dist)
    plt.hist(flat_mat, bins=max_dist, rwidth=0.9)
    plt.savefig('./data/fig/length_distribution.png')
    plt.close()

    return max_dist, dist_mat


def struc2vec(G, k, distance_func='pad_norm', model='comp', view_arg=[(5, 0.5)], drop_prop=0.5, scale=4, is_weight=True):
    print('using', distance_func, 'to calculate distance matrix')
    node_num = G.number_of_nodes()
    # reconstrcut adj matrix
    adj_mat = G.adj().to_dense()

    # calculate nodesdegree
    if not is_weight:
        nodes_degree = G.in_degrees().numpy()
    else:
        edge_weights = G.edata['w'].numpy()
        # build weighted matrix
        h, t = G.all_edges(order='eid')
        weight_adj = np.zeros((node_num, node_num))
        weight_adj[h.numpy(), t.numpy()] = edge_weights
        nodes_degree = np.sum(weight_adj, axis=0)
        # print(nodes_degree)

    # check_para(nodes_degree)
    def k_distance_set_tensor(node, k, is_weight=True):
        node_list = []
        # initialization
        check_vec = torch.ones(node_num)
        N_k = torch.zeros(node_num)
        N_k[node] = 1
        check_vec[node] = 0
        node_list.append([node])

        # find k neighbor iteratively
        for i in range(k):
            N_k_tmp = torch.matmul(adj_mat, N_k)
            N_k = check_vec * N_k_tmp
            # regulize nk
            regular = torch.where(N_k == 0, torch.tensor(1.), N_k)

            N_k = N_k / regular
            check_vec = check_vec - N_k

            node_list.append(list(torch.nonzero(N_k).T.numpy()[0]))

            # fill degree list
            degree_seq = []
            for nodes in node_list:
                single_degree_list = [nodes_degree[dgr] for dgr in nodes]
                # single_degree_list = []
                single_degree_list.sort(reverse=True)
                degree_seq.append(single_degree_list)
        return degree_seq

    def k_distance_set(node, k):
        t0 = time.time()
        node_list = []
        # k=0 self
        node_list.append([node])

        for i in range(1, k + 1):
            base_nodes = node_list[i - 1]
            check_list = []
            for c in range(i):
                check_list.extend(node_list[c])

            #check_para(check_list)
            neighbor_set = []

            get_nlist0 = time.time()
            for node in base_nodes:
                neighbor_set.extend([idx for idx, wgt in enumerate(adj_mat[node]) if wgt == 1 and idx not in check_list])

            neighbor_set.sort()
            node_list.append(neighbor_set)
            get_nlist1 = time.time()
            #print('get ', i, ' layer neighbor list time ', get_nlist1 - get_nlist0)

        #check_para(node_list)
        degree_seq = []
        for nodes in node_list:
            degree_seq.append([nodes_degree[dgr] for dgr in nodes])

        #check_para(degree_seq)
        t1 = time.time()
        #print('get ', node, ' degree seq time ', t1 - t0)
        return degree_seq

    degree_seqs = []
    # generate degree seq for all nodes
    # in this list k=0 is degree of it self
    for node in range(node_num):
        degree_seq = k_distance_set_tensor(node, k)
        # print(node, degree_seq)
        degree_seqs.append(degree_seq)

    # setup function for distance matrix compute
    if distance_func == 'DTW':
        func_g = DTW
    elif distance_func == 'pad_norm':
        func_g = pad_norm

    # calculate strcuture similarity
    similarity_mat = torch.zeros((node_num, node_num, k + 1))
    if model == 'normal':
        weight_mat = torch.zeros((node_num, node_num, k + 1))
    elif model == 'comp':
        weight_mat = torch.zeros((node_num, node_num))

    # set layer weights to decide importance of each layers
    # weights function pdf of normal distribution
    def normal_pdf(x, e=0, v=1):
        return 1 / (sqrt(2 * PI) * v) * exp(0 - pow(x - e, 2) / (2 * v * v))

    X = [i for i in range(k + 1)]
    layer_weights = []
    for view in view_arg:
        e_vec = [view[0] for i in range(k + 1)]
        v_vec = [view[1] for i in range(k + 1)]

        layer_weights.append(list(map(normal_pdf, X, e_vec, v_vec)))

    layer_weights = np.sum(np.array(layer_weights), axis=0)
    check_para(layer_weights)

    for i in range(node_num):
        for j in range(node_num):
            for hop in range(0, k + 1):
                if hop == 0:
                    similarity_mat[i][j][hop] = func_g(degree_seqs[i][hop], degree_seqs[j][hop])
                else:
                    if degree_seqs[i][hop] and degree_seqs[j][hop]:
                        similarity = func_g(degree_seqs[i][hop], degree_seqs[j][hop])  # + similarity_mat[i][j][hop - 1]
                        similarity_mat[i][j][hop] = similarity
                        #print(similarity)\
                        if model == 'normal':
                            if i == j:
                                weight_mat[i][j][hop] = 0
                            else:
                                weight_mat[i][j][hop] = layer_weights[hop] * exp(-similarity)

    # scale to 0 and 4
    scaler = similarity_mat.max() / scale
    for i in range(node_num):
        for j in range(node_num):
            if model == 'comp':
                layer_sim = torch.exp(-similarity_mat[i][j][:] / scaler)
                layer_weights = torch.tensor(layer_weights).to(dtype=torch.float)
                weight_mat[i][j] = torch.matmul(layer_sim, layer_weights)

                #print(weight_mat[i][j])
                if i == j:
                    weight_mat[i][j] = 0

    # initialization
    head = []
    tail = []
    edge_weights = []
    MG = dgl.DGLGraph()

    #check_para(weight_mat)
    #print(torch.nonzero(weight_mat).size())

    def proportional_drop(ori_weight_mat, prop=0.8):
        # return the threshold needed
        prop_idx = int(node_num * node_num * prop)
        weights, indices = torch.sort(weight_mat.view(-1))
        prop_thresh = weights.numpy()[prop_idx]

        res_weight_mat = torch.where(ori_weight_mat < prop_thresh, torch.zeros((node_num, node_num)), ori_weight_mat)

        return res_weight_mat

    weight_mat = proportional_drop(weight_mat, drop_prop)
    print('maintained edges', torch.count_nonzero(weight_mat))

    weight_mat = weight_mat.numpy()
    if model == 'comp':
        # build compressed graph
        # considering dropout certain percentage
        MG.add_nodes(node_num)
        for i in range(node_num):
            for j in range(node_num):
                #print(weight_mat[i][j])
                if weight_mat[i][j] > 0:
                    #print(i, j, weight_mat[i][j])
                    head.append(i)
                    tail.append(j)
                    edge_weights.append(weight_mat[i][j])
                    #print(degree_seqs[i])
                    #print(degree_seqs[j])
                else:
                    #print(degree_seqs[i])
                    #print(degree_seqs[j])
                    pass

        #print(len(head))
        MG.add_edges(head, tail)
        MG.edata['w'] = torch.tensor(edge_weights)
        MG = dgl.add_reverse_edges(MG, copy_edata=True)

        #print(MG.adj())

        export_adj_mat = MG.adj().to_dense().numpy()
        np.savetxt('./data/fig/dist_mat.csv', export_adj_mat, delimiter=',')

    elif model == 'normal':
        # build multi layer graph
        MG.add_nodes(node_num * k)

        for layer in range(k + 1):
            for i in range(node_num):
                for j in range(node_num):
                    tmp_weight = weight_mat[i][j][layer]
                    if tmp_weight != 0:
                        head.append(i + layer * node_num)
                        tail.append(j + layer * node_num)
                        edge_weights.append(tmp_weight)

        MG.add_edges(head, tail)
        MG.edata['w'] = torch.tensor(edge_weights)
        MG = dgl.add_reverse_edges(MG, copy_edata=True)

        # connect layers
        layer_edge_head = []
        layer_edge_tail = []
        layer_edge_weights = []
        # print(weight_mat)
        lyr_weights_avg = np.sum(np.sum(weight_mat, axis=0), axis=0) / (node_num * node_num - node_num)

        for layer in range(k + 1):
            for i in range(node_num):
                index = i + layer * node_num
                if layer == 0:
                    layer_edge_head.append(i)
                    layer_edge_tail.append(i + node_num)
                    up_weight = log(sum([1 for node in weight_mat[i] if node[layer] >= lyr_weights_avg[layer]]) + exp(1))
                    layer_edge_weights.append(up_weight)
                elif layer == k:
                    layer_edge_head.append(index)
                    layer_edge_tail.append(index - node_num)
                    layer_edge_weights.append(1)
                else:
                    layer_edge_head.append(index)
                    layer_edge_tail.append(index - node_num)
                    layer_edge_weights.append(1)
                    layer_edge_head.append(index)
                    layer_edge_tail.append(index + node_num)
                    up_weight = log(sum([1 for node in weight_mat[i] if node[layer] >= lyr_weights_avg[layer]]) + exp(1))
                    layer_edge_weights.append(up_weight)

        #show_normal_graph(G)

        # connect layers together
        layer_edge_weights = torch.tensor(layer_edge_weights)
        tmp_weight = torch.cat((MG.edata['w'], layer_edge_weights), dim=0)

        MG.add_edges(layer_edge_head, layer_edge_tail)
        MG.edata['w'] = tmp_weight

    nx_MG = MG.to_networkx()
    nx_MG = nx_MG.to_undirected()
    print('Graph connectivity: ', nx.is_connected(nx_MG))

    return MG
