"""
learning structure from data
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
from data_manger.utilities import get_file_list
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import heapq
from math import log


k = 2

#priori knowledge
K = np.array([[1]*5]*5)
for i in range(5):
    K[i, i] = 0
K[0, 1] = 0
K[1, 0] = 0
K[0, 2] = 0
K[2, 0] = 0
K[1, 2] = 0
K[2, 1] = 0

#prepare data
DATA_PATH = parentdir + "\\bpsk_navigate\\data\\test\\"
list_files = get_file_list(DATA_PATH)
graph = 0
for i, file in enumerate(list_files):
    data = np.load(DATA_PATH + file)
    split = int(len(data) / 2)
    normal = data[:split,:]
    fault = data[split:,:]
    df0 = pd.DataFrame(normal)
    df1 = pd.DataFrame(fault)
    tmp0 = df0.corr('spearman')
    tmp1 = df1.corr('spearman')
    tmp = (tmp0 + tmp1 )/2
    graph = (graph * (i/(i + 1))) + tmp / (i+1)

graph = abs(graph)
graph = K*graph
graph = graph.values

#choose the K best neighbors
thresh = np.zeros(len(graph))
for i in range(len(graph)):
    nb_i = graph[i, :]
    nbs = heapq.nlargest(k, nb_i)
    nbs = nbs[-1]
    thresh[i] = (nbs)

G = np.zeros((len(graph), len(graph)))
for i in range(len(graph)):
    for j in range(i+1, len(graph)):
        corr = float(graph[i, j])
        if corr >= thresh[i]:
            G[i, j] = corr
            G[j, i] = corr

#delete some edges
for i in range(len(graph)):
    nb_i = G[i, :].copy()
    if np.sum(nb_i > 0) > k:#too many neighbors
        min_nb = np.where()


#---------------------function definitions--------------------#
def connected_with_out(graph, i, j):
    """
    check if it is connected graph without i-j
    """
    G = (graph > 0)
    G[i, j] = False
    G[j, i] = False
    visited = set()
    tobe = set()
    tobe.add(0)
    while tobe:
        tmp_tobe = set()
        for i in tobe:
            visited.add(i)
            nb = G[i, :]
            nb = np.where(nb)
            for n in nb:
                if n not in visited:
                    tmp_tobe.add(n)
        tobe = tmp_tobe
    
    for i in range(len(graph)):
        if i not in visited:
            return False
    return True
