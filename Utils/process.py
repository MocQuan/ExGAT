import json
from multiprocessing import Condition
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import scipy.io as scio
import pandas as pd
from collections import defaultdict
import math
import random

from utils import ppmi

"""
Some para
"""
R_c = 0.5


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node. 
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)
    # return 0 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # # PageRank Enhance
    # if pr:
    #     print('Enhanced by PageRank')
    #     graph = enhance_adj_pagerank(graph, 30, 2)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_pr(dataset_str, num_node=0, khop=2, pr=True, cond=0):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    count = 0
    # PageRank Enhance
    if pr:
        print('Enhanced by PageRank')
        adj, count = enhance_adj_pagerank(graph, num_node, khop, cond=cond)
    else:
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).todense()
    # print(type(nx.adjacency_matrix(nx.from_dict_of_lists(graph))))
    # print(type(sp.csr_matrix(adj)))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    print(labels[2407])
    print(labels[0])
    print(labels[2338])

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, count




def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack(
        (adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape


def enhance_adj_pagerank(graph, num, depth, dump=0.85, cond=0):
    G = nx.from_dict_of_lists(graph)
    pr = nx.pagerank(G, dump)
    pr_usort = pr.items()
    pr_sort = sorted(pr_usort, key=lambda s: s[1], reverse=True)
    # print(pr_sort)
    # print(pr[0])
    adj = nx.adjacency_matrix(G).todense()
    # print(adj.shape)
    # print(adj)

    # condition define
    if cond == 4:
        condition = [1, 2]
        print('Degree+Connection')
    elif cond == 5:
        condition = [2, 3]
        print('Connection+Locality')
    elif cond == 6:
        condition = [1, 2, 3]
        print('Degree+Connection+Locality')
    else:
        if cond == 1:
            print('Degree-Aware')
        if cond == 2:
            print('Connection-Aware','R_C: ', R_c)
        if cond == 3:
            print('Locality-Aware')
        if cond == 9:
            print('New Locality-Aware')
        condition = [cond]
        print(condition)
    # condition = [1, 2]
    # average_d = 0
    if 1 in condition:
        D = nx.degree(G)
        count = 0
        for i in D:
            count += i[1]
            # d_min = min(d_min, i[1])
            # d_max = max(d_max, i[1])
        average_d = sum(i[1] for i in D) / len(nx.nodes(G))
        # d_min = min(i[1] for i in D)
        # d_max = max(i[1] for i in D)
        # number of d(node) < average_d
        print('number of d(node) < average_d', sum(i[1] < average_d for i in D))

    list_min_pr_nei = []
    list_pr_local = []
    store_flag = True
    # Only mode 3 need pre_store
    if 3 in condition:
        try:
            store_mat = scio.loadmat('./PR/' + str(len(graph)) + '_' + str(depth)+ str(cond) + '.mat')
            list_min_pr_nei = store_mat.get('prmin').tolist()
            list_pr_local = store_mat.get('prlocal')[0].tolist()
        except IOError:
            store_flag = False

    # select top node
    if num < 1:
        num = round(len(pr_sort) * num)
    else:
        num = min(len(pr_sort), num)

    pr_select = []
    for idx, _ in pr_sort[:num]:
        pr_select.append(idx)
    count = 0
    involve = 0
    for i in graph:
        nei_ = get_neigbors(G, i, depth)
        nei = []
        for idx_nei in nei_:
            nei.extend(nei_[idx_nei])

        # enhance local
        if 3 in condition:
            if store_flag:
                min_pr_nei = list_min_pr_nei[i]
                pr_local = list_pr_local[i]
            else:
                local_hop = depth + 1
                H = nx.ego_graph(G, i, local_hop, True, True)
                pr_local = nx.pagerank(H, dump)
                pr_imme_nei = [[node_nei, pr_local[node_nei]] for node_nei in graph[i]]
                pr_imme_nei.append([i, pr_local[i]])
                min_pr_nei = min(pr_imme_nei, key=lambda s: s[1])
                list_min_pr_nei.append(min_pr_nei)
                list_pr_local.append(list(pr_local.items()))

        if 9 in condition:
            local_rate=0.8
            pr_imme_nei = [pr[node_nei] for node_nei in graph[i]]
            pr_imme_nei_sort = sorted(pr_imme_nei)
            pr_loc=pr_imme_nei_sort[math.ceil(len(pr_imme_nei)*local_rate)-1]

        involve_each = 0
        for j in pr_select:
            if j in nei and j not in graph[i]:
                # print(j)
                if j == i:
                    continue
                if 1 in condition:
                    # print('Degree-Aware')
                    # if len(graph[i]) > average_d:
                    r_d = 0.4
                    d_sort = sorted([d for n, d in G.degree()], reverse=False)
                    thresh_d = d_sort[int(r_d * len(d_sort))]
                    # Largest one
                    # thresh_d = d_sort[-1]
                    if len(graph[i]) >= thresh_d:
                        continue
                if 2 in condition:
                    # print('Connection-Aware')
                    count_condition = 0
                    for n_condition in graph[i]:
                        if j in graph[n_condition]:
                            count_condition += 1
                    if count_condition / len(graph[i]) <= R_c:
                        continue
                if 3 in condition:
                    # print('Locality-Aware')
                    if store_flag:
                        for pr_k, pr_v in pr_local:
                            if pr_k == j:
                                pr_local_j = pr_v
                    else:
                        pr_local_j = pr_local[j]
                    if pr_local_j < min_pr_nei[1]:
                        print('min_pr_nei', min_pr_nei)
                        print('j', j)
                        print('--------')
                        continue
                if 9 in condition:
                    if pr[j] < pr_loc:
                        continue
                count += 1
                adj[i, j] = 1
                involve_each = 1
                # adj[j, i] = 1
        involve += involve_each
    if not store_flag:
        scio.savemat('./PR/' + str(len(graph)) + '_' + str(depth) + '_' + str(cond) + '.mat',
                     mdict={'prmin': list_min_pr_nei, 'prlocal': list_pr_local})
        print('Success Save !')
    if 1 in condition:
        print('thresh_d', thresh_d)
    print("Enhance num: " + str(count))
    print("Involve num: " + str(involve))
    # return
    return adj, count

def get_neigbors(g, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output



def load_socialdata_pr(dataset_str, num_node=0, khop=2, sp_strategy=-1, pr=True,
                       cond=0):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    print('social dataset')
    data = scio.loadmat('./data/' + dataset_str)

    adj = data.get('Network')

    Feature = data.get('Attributes')

    # 读取Label，并转化为onehot的形式
    labels = data.get('Label')
    labels_ = labels.copy()
    one_hot_label = np.zeros(shape=(labels.shape[0], len(np.unique(labels))))  ##生成全0矩阵
    for i in range(len(one_hot_label)):
        one_hot_label[i, labels[i] - 1] = 1
    labels = one_hot_label
    # print(labels)
    features = Feature.tolil()

    indices = np.arange(adj.shape[0]).astype('int32')
    np.random.seed(999)
    np.random.shuffle(indices)

    if sp_strategy > 0:
        # ----------------------------------------------
        # per node
        print('Per Node Sample')
        per_num = sp_strategy
        sample_devide_class = [[], [], [], [], [], []]
        for i in indices:
            sample_devide_class[int(labels_[i]) - 1].append(i)

        idx_train = []
        for i in range(len(sample_devide_class)):
            np.random.shuffle(sample_devide_class[i])
            idx_train.extend(sample_devide_class[i][:per_num])

        idx_val = indices[-1500:-1000]
        idx_test = indices[-1000:]

    elif sp_strategy == -1:
        # ------------------------------------------------
        # 333
        print("Proportional Division")
        idx_train = indices[:adj.shape[0] // 3]
        idx_val = indices[adj.shape[0] // 3: (2 * adj.shape[0]) // 3]
        idx_test = indices[(2 * adj.shape[0]) // 3:]
    else:
        # 127
        print('Standard Division')
        train_size = (1 * adj.shape[0]) // 10
        val_size = (2 * adj.shape[0]) // 10
        test_size = (7 * adj.shape[0]) // 10

        idx_train = indices[:train_size]
        idx_val = indices[train_size: val_size + train_size]
        idx_test = indices[val_size + train_size:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # load graph from DF
    # ran = range(adj.shape[0])
    # DF_adj = pd.DataFrame(adj.toarray().astype(np.int8), index=ran, columns=ran)
    # # print(DF_adj)
    # G = nx.from_pandas_adjacency(DF_adj)
    # graph = nx.to_dict_of_lists(G)

    # load graph from DefaultDictionary
    adj = adj.toarray()
    print(adj.shape)
    graph = defaultdict(list)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] > 0:
                graph[i].append(j)
    count = 0
    # PageRank Enhance
    if pr:
        print('Enhanced by PageRank')
        adj, count = enhance_adj_pagerank(graph, num_node, khop, cond=cond)
    else:
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).todense()

    print(adj.shape)
    print(features.shape)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, count


