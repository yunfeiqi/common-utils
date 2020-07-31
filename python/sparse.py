#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description    :
    Util Functions for Sparse Matirx
@Time    :2020/07/23 17:00:03
@Author    :sam.qi
@Version    :1.0
'''


import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import diags

def search_node_by_degree(adj, degree=100):
    '''
    @Description    :
        Search Graph Node that with specific degree value
    @Time    :2020/07/31 14:56:52
    @Author    :sam.qi
    @Param    : 
        adj: sparse adjacent matrix for graph
        degree: specific degree
    @Return    :
        target node index that degree equal specific value
    '''
    
    
    node_degree = adj.sum(axis=0)
    node_degree = np.array(node_degree.tolist())
    target_idx = list(np.where(node_degree == degree)[1])
    return target_idx


def average_node_feature(adj, features):
    '''
    @Description    :
        Compute average feature base on current node and neighbor node
    @Time    :2020/07/31 15:07:23
    @Author    :sam.qi
    @Param    :
        adj: Graph adjacent matrix
        features: node feature's matrix
    @Return    :
        average feature
    '''
    
    
    neighbor_features = adj.dot(features)
    node_degree = adj.sum(axis=0)
    node_degree = np.array(node_degree.tolist())
    node_degree = node_degree.T

    avg_neighbor_feature = neighbor_features / node_degree
    return avg_neighbor_feature


def reset_sparse(adj, clear_nodes):
    '''
    @Description    : 清除邻接矩阵节点
        1. 构建对角阵 A
        2. 将 clear_nodes 对应位置致0
        3. 对角阵A 乘以 adj 清空 adj 对应行
        4. adj 转置
        5. 对角阵A 乘以 adj 清空 adj 对应列
        5. adj 转置，回复原来顺序
    @Time    :2020/07/23 18:53:21
    @Author    :sam.qi
    @Param    :
    @Return    :

    '''
    # 构建对角阵
    shape = adj.shape
    row_size = shape[0]
    diag_row = list(range(row_size))
    diag_col = list(range(row_size))
    diag_data = [1]*row_size

    for n_idx in clear_nodes:
        diag_data[n_idx] = 0

    diag_M = coo_matrix((diag_data, (diag_row, diag_col)), shape=shape)

    # 清空行和列
    r_clear_adj = diag_M.dot(adj)
    r_c_clear_adj = diag_M.dot(r_clear_adj.T)

    # 恢复
    new_adj = r_c_clear_adj.T
    new_adj = new_adj.tocsr()

    return new_adj

def set_diag(adj):
    '''
    @Description    :
        Set main diag with 0 
    @Time    :2020/07/31 15:12:45
    @Author    :sam.qi
    @Param    :
    @Return    :
    '''

    adj = adj - diags(adj.diagonal())
    
    return adj 
