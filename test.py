import googlemaps
from datetime import datetime
import pandas as pd
import numpy as np
import networkx as nx
import tools
import dgl
import torch
from math import sqrt, exp
from math import pi as PI
import matplotlib.pyplot as plt
import sys
import csv

import label_propogation as lp
from GCN import *
from graph_gen import *
from data_processing import convert_MTS_imge, cut_weeks, reform_data, concat_acndata, window_slide
import data_processing as dp
import graph_gen
import data_processing as dp
import graph_gen as gg
import tools


def plot_acc_node_num():
    node_nums = [i for i in range(100, 2001, 50)]
    ncr_acc_node = []
    with open('./data/record_data/toy_node_sslp.txt', 'r', encoding='utf-8') as f:
        for item, node_num in zip(f.readlines(), node_nums):
            #print(node_num, float(item.split('[')[1].split(']')[0]))
            ncr_acc_node.append(float(item.split('[')[1].split(']')[0]))

    toy_acc_node = []
    with open('./data/record_data/toy_node_lp.txt', 'r', encoding='utf-8') as f:
        for item, node_num in zip(f.readlines(), node_nums):
            #print(node_num, float(item.split('[')[1].split(']')[0]))
            toy_acc_node.append(float(item.split('[')[1].split(']')[0]))

    node_nums = np.array(node_nums)
    ncr_acc_node = np.array(ncr_acc_node)
    toy_acc_node = np.array(toy_acc_node)

    plt.figure(figsize=(8, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(node_nums, ncr_acc_node, marker='o', color="green", label="SSLP", linewidth=1.5)
    plt.plot(node_nums, toy_acc_node, marker='o', color="red", label="LP", linewidth=1.5)

    plt.title('Accuracy variation according to node number on Toy Example')
    plt.xlabel('number of nodes')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()


def plot_ncr_node_num():
    node_nums = [i for i in range(100, 1900, 50)]
    node_nums.append(1871)
    print(node_nums)
    sslp_acc_node = []
    with open('./data/record_data/ncr_node_sslp.txt', 'r', encoding='utf-8') as f:
        for item, node_num in zip(f.readlines(), node_nums):
            sslp_acc_node.append(float(item.split('[')[1].split(']')[0]))

    lp_acc_node = []
    with open('./data/record_data/toy_node_lp.txt', 'r', encoding='utf-8') as f:
        for item, node_num in zip(f.readlines(), node_nums):
            lp_acc_node.append(float(item.split('[')[1].split(']')[0]))

    node_nums = np.array(node_nums)
    sslp_acc_node = np.array(sslp_acc_node)
    lp_acc_node = np.array(lp_acc_node)

    plt.figure(figsize=(8, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(node_nums, sslp_acc_node, marker='o', color="green", label="SSLP", linewidth=1.5)
    plt.plot(node_nums, lp_acc_node, marker='o', color="red", label="LP", linewidth=1.5)

    # plt.xticks()
    plt.title('Accuracy variation according to node number on NCR data')
    plt.xlabel('number of nodes')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()
    return


def plot_acc_view():
    view = [int(i + 1) for i in range(21)]

    with open('./data/record_data/toy_view.txt', 'r', encoding='utf-8') as f:
        ncr_data = [float(i) for i in f.readlines()[0].split(',')]
        baseline = [0.69625 for i in range(21)]

    node_nums = np.array(view)
    ncr_acc_node = np.array(ncr_data)
    toy_acc_node = np.array(baseline)

    plt.figure(figsize=(8, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(node_nums, ncr_acc_node, marker='o', color="green", label="SSLP", linewidth=1.5)
    plt.plot(node_nums, toy_acc_node, color="red", label="LP", linewidth=1.5)

    plt.xticks(view, view)
    plt.title('Accuracy variation according to views on Toy Example')
    plt.xlabel('view')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()


def plot_ncr_view():
    view = [int(i + 1) for i in range(18)]
    baseline = [0.6723 for i in range(18)]
    # ncr_data = [0.69625 for i in range(21)]
    ncr_data = []
    with open('./data/record_data/ncr_view.txt', 'r', encoding='utf-8') as f:
        for item, node_num in zip(f.readlines(), view):
            #print(node_num, float(item.split('[')[1].split(']')[0]))
            ncr_data.append(float(item.split(']')[1]))

    print(ncr_data)

    node_nums = np.array(view)
    ncr_acc_node = np.array(ncr_data)
    toy_acc_node = np.array(baseline)

    plt.figure(figsize=(8, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(node_nums, ncr_acc_node, marker='o', color="green", label="SSLP", linewidth=1.5)
    plt.plot(node_nums, toy_acc_node, color="red", label="LP", linewidth=1.5)

    plt.xticks(view, view)
    plt.title('Accuracy variation according to views on NCR data')
    plt.xlabel('view')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()


def plot_acn_view():
    view = [int(i + 1) for i in range(17)]

    with open('./data/record_data/acn_view.txt', 'r', encoding='utf-8') as f:
        ncr_data = [float(i) for i in f.readlines()[0].split(',')][:17]
        baseline = [0.49206349206349204 for i in range(17)]

    print(len(ncr_data))
    print(len(view))

    node_nums = np.array(view)
    ncr_acc_node = np.array(ncr_data)
    toy_acc_node = np.array(baseline)

    plt.figure(figsize=(8, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(node_nums, ncr_acc_node, marker='o', color="green", label="SSLP", linewidth=1.5)
    plt.plot(node_nums, toy_acc_node, color="red", label="LP", linewidth=1.5)

    plt.xticks(view, view)
    plt.title('Accuracy variation according to views on ACN data')
    plt.xlabel('view')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()


def plot_acn_node_num():
    node_nums = [i for i in range(100, 501, 50)]
    print(node_nums)
    sslp_acc_node = []
    with open('./data/record_data/acn_node_sslp.txt', 'r', encoding='utf-8') as f:
        for item, node_num in zip(f.readlines(), node_nums):
            sslp_acc_node.append(float(item.split('[')[1].split(']')[0]))

    lp_acc_node = []
    with open('./data/record_data/acn_node_lp.txt', 'r', encoding='utf-8') as f:
        for item, node_num in zip(f.readlines(), node_nums):
            lp_acc_node.append(float(item.split('[')[1].split(']')[0]))

    node_nums = np.array(node_nums)
    sslp_acc_node = np.array(sslp_acc_node)
    lp_acc_node = np.array(lp_acc_node)

    plt.figure(figsize=(8, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(node_nums, sslp_acc_node, marker='o', color="green", label="SSLP", linewidth=1.5)
    plt.plot(node_nums, lp_acc_node, marker='o', color="red", label="LP", linewidth=1.5)

    # plt.xticks()
    plt.title('Accuracy variation according to node number on NCR data')
    plt.xlabel('number of nodes')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()
    return


#plot_ncr_node_num()
# plot_acc_node_num()
plot_acc_view()
plot_ncr_view()
# plot_acn_node_num()
plot_acn_view()