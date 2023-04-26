from csv import writer
import googlemaps
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import tools
import dgl
import torch
from math import sqrt, exp
from math import pi as PI
import sys

#from data_processing import *
#from label_propogation import *
from GCN import *
# from graph_gen import *

import GCN
import data_processing as dp
import label_propogation as lp
import graph_gen as gg
import visualization as viz

# -----------------------------------------------------
dataset_out_path = './data/acn_data/mts_data.csv'
recorder = tools.file_wirter('./record/')


def train(dataset='ncr_data', propagation_method='s2v', node_num=800):
    try:
        if dataset == 'toy':
            print('load toy example')
            graph_data, labels = gg.generate_test_set(node_num, load=True)
            dist_mat = gg.distance(graph_data)
            #G = gg.kNN(dist_mat)
            #G = gg.set_labels(G, labels, False)
            G = gg.real_graph(graph_data, True)
            data = graph_data[:, 1:]

        elif dataset == 'acn_data':
            print('load acn_data')
            data = []

            data1 = pd.read_csv('./data/acn_data/acndata.csv')
            data2 = dp.reform_data(data1)
            data3, labels = dp.window_slide(data2, node_num=node_num, step_len=5)

            graph_data = dp.convert_MTS_imge(data3, labels, dataset_out_path)
            #labels = np.array(labels)

            G = gg.real_graph(graph_data, True)

        elif dataset == 'ncr_data':
            print('load ncr_data')
            graph_data, labels = dp.ncr_data(node_num=node_num)
            labels = np.array(labels)
            G = gg.real_graph(graph_data, True)
            data = graph_data[:, 1:]

        nx_MG = G.to_networkx()
        nx_MG = nx_MG.to_undirected()

        print('sample number: ', G.number_of_nodes())
        print('Graph connectivity: ', nx.is_connected(nx_MG))

        print('load data done')
        ori_label = G.ndata['L']
        viz.show_graph(G, data=data, save_fig=True, name='origin')
        print(G.edges()[0].size())

        print('nodes number: ', G.number_of_nodes())
        # print('Original modularity: ', lp.calculate_modularity(G))

        rng = [i / 10 for i in range(11)]
        r = 0.5

        #tools.reset_file('./data/fig/acc_data.txt')

        # record information
        modularity_list = []
        acc_list = []

        if propagation_method == 's2v':
            view_list = [i for i in range(1, 20)]
            # view_list = [8]
        elif propagation_method == 'normal':
            view_list = [8, 9, 10]

        for i in view_list:
            view = [(i - 1, 2), (i, 0.5), (i + 1, 2)]
            # view = [(i, 2)]

            print(view)
            # G = kNN(dist_mat)
            G = gg.real_graph(graph_data, False)
            print(G.edges()[0].size())

            viz.show_graph(G, data, True, 'seq')
            # G = label_propagation(G, emb='normal', epoch_num=0)
            G = lp.label_propagation(G, emb=propagation_method, threshold=0.1, epoch_num=100, view_arg=view, data=data)

            # regain labels
            regain_labels = torch.where(G.ndata['L'] == 1)[1].numpy()
            comp = labels == regain_labels

            acc = sum(comp) / len(regain_labels)
            print('accuracy: ', acc)
            #tools.write_numeric('./data/fig/acc_data_view.txt', [view, acc])
            recorder.write_list('acc_with_view.txt', [view, acc])

            acc_list.append(acc)
            #modularity = lp.calculate_modularity(G)
            #modularity_list.append(modularity)

        #print(modularity_list)
        print(acc_list)

        return acc_list

    except OverflowError:
        print('wrong')
        main()
        return


def main():
    acc_list = []
    node_nums = [i for i in range(100, 501, 50)]
    node_nums = [800]
    for node_num in node_nums:  # range(100, 101, 50):
        print('number of nodes in graph {}'.format(node_num))
        acc = train(dataset='toy', propagation_method='normal', node_num=node_num)
        #tools.write_numeric('./data/fig/acc.txt', [node_num, acc])
        recorder.write_list('acc_with_node_num.txt', [node_num, acc])
        acc_list.extend(acc)

    tools.write_numeric('./data/fig/acc_data.txt', acc_list)


if __name__ == "__main__":
    main()
