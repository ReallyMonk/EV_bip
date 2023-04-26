import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import dgl
import dgl.function as fn
from dgl.nn.pytorch import conv
from dgl.data import citation_graph
from dgl.nn.pytorch.conv import GraphConv
import networkx as nx

dgl.load_backend('pytorch')
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class nodeModule(nn.Module):

    def __init__(self, in_fea, out_fea, activation):
        super(nodeModule, self).__init__()
        self.linear = nn.Linear(in_fea, out_fea)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation:
            h = self.activation(h)

        return {'h': h}


class GCModule(nn.Module):

    def __init__(self, g, in_fea, out_fea, activation, dropout):
        super(GCModule, self).__init__()
        self.g = g
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            dropout = 0.

        self.node_update = nodeModule(in_fea, out_fea, activation)

    def forward(self, feature):
        #print(feature.size())
        self.g.ndata['h'] = feature
        self.g.update_all(gcn_msg, gcn_reduce, self.node_update)
        #g.apply_ndoes(func=self.node_update)
        return self.g.ndata.pop('h')


class GCN(nn.Module):

    def __init__(self, g, in_fea, out_fea, hidden, k):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GCModule(g, in_fea, hidden, F.relu, 0.))

        dropout = 0.5

        for i in range(k - 1):
            self.layers.append(GCModule(g, hidden, hidden, None, dropout))

        self.layers.append(GCModule(g, hidden, out_fea, None, dropout))

    def forward(self, features):
        x = features
        for layer in self.layers:
            #print('-' * 10)
            x = layer(x)
        #x = self.gcn2(g, x)

        return x


def compute_filter(g, k=1, alpha=1, weight=True):
    node_num = g.number_of_nodes()
    adj_mat = g.adj()

    degrees = g.in_degrees()

    # # build weighted adj matrix if there is matrix
    if weight:
        # adj matrix
        node_num = g.num_nodes()
        indices = torch.tensor(
            np.array([g.edges()[0].numpy(),
                      g.edges()[1].numpy()]))
        weights = g.edata['w'].float()
        size = torch.Size((node_num, node_num))
        adj_mat = torch.sparse.FloatTensor(indices, weights, size).to_dense()

        # print(adj_mat)
        # degree matrix
        degrees = torch.sum(adj_mat, dim=1)

    else:
        adj_mat = adj_mat.to_dense()

    d_mat = torch.diag(degrees)
    lap_mat = d_mat - adj_mat
    #print(lap_mat)
    #print(lap_mat)

    # print(torch.matmul(input, other))
    # column is the corresponding eigen vectors
    eigvalues, eigvectors = torch.linalg.eig(lap_mat)
    eigvalues, eigvectors = eigvalues.real, eigvectors.real.transpose_(0, 1)

    #print(eigvalues, eigvectors)
    #print(torch.mm(eigvectors, eigvectors.transpose_(0, 1)))

    # sysmetric normalized
    sy_degree = torch.diag(torch.pow(degrees, -0.5))
    #print(degrees)
    #print(sy_degree)
    sym_L = torch.eye(node_num) - torch.mm(torch.mm(sy_degree, adj_mat),
                                           sy_degree)

    #print(sym_L)
    # graph filter I-sym_L
    graph_filter = torch.eye(node_num) - alpha * sym_L

    for i in range(k - 1):
        graph_filter = torch.mm(graph_filter, graph_filter)

    #print(eigvalues)
    #print(eigvectors)
    return graph_filter


def node_similarity(g, h):
    # update edge weight with embedding nodes features
    # we want to make larger weights represents stronger connection -log(||x_i-x_j||_2)
    origin_adj_mat = g.adj().coalesce()
    indices = origin_adj_mat.indices().numpy()
    size = origin_adj_mat.size()

    weights = []

    for idx, (i, j) in enumerate(zip(indices[0], indices[1])):
        dist = torch.norm(h[i] - h[j], p=2)
        weight = 1 - torch.log(dist)
        weights.append(weight)

    indices = torch.tensor(indices)
    weights = torch.FloatTensor(weights)
    weight_adj = torch.sparse.FloatTensor(indices, weights, size).to_dense()

    #print(weight_adj)
    return weight_adj


def load_cora_data():
    data = citation_graph.load_cora('./test_graph_data/cora')
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = dgl.DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train():
    g, features, labels, train_mask, test_mask = load_cora_data()

    gcn = GCN(g, 1433, 32, 32, 3)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-3)
    dur = []
    for epoch in range(50):
        if epoch >= 3:
            t0 = time.time()

        gcn.train()
        logits = gcn(features)
        print(logits.size())
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(gcn, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".
              format(epoch, loss.item(), acc, np.mean(dur)))
