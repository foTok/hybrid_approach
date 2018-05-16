"""
analyze feature relationship
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Digraph
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.Bayesian_learning import Bayesian_structure
from graph_model.Bayesian_learning import Bayesian_learning
from graph_model.utilities import organise_data

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
FE_PATH = PATH + "\\ann_model\\"
fe_file = "FE0.pkl"
step_len=100
batch = 20000

#load fe
FE = torch.load(FE_PATH+fe_file)
FE.eval()

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

#labels, 6; features, 10 (12-10); residuals 3.
BL = Bayesian_learning(6+10+3)
BL.init_queue()
inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)

df = pd.DataFrame(batch_data)
# pearson
relation_graph = df.corr()
relation_graph = abs(relation_graph)

# np.savetxt("relation.txt", relation_graph)
# print(relation_graph)

# no self connection, relation_graph[i,i]=0 (i=0,1...18)
for i in range(19):
    relation_graph[i][i] = 0

# no edges between fault, relation_graph[i,j]=0 (i,j=0,1...5)
for i in range(6):
    for j in range(i+1, 6):
        relation_graph[i][j] = 0
        relation_graph[j][i] = 0

# all edges from fault to features, relation_graph[i,j]=1. (i/j=0,1...5, j/i=6,7...18)
for i in range(6):
    for j in range(6, 19):
        relation_graph[i][j] = 1
        relation_graph[j][i] = 1

# no edges between features with correfficency < 0.6
for i in range(6, 19):
    for j in range(i+1, 19):
        if relation_graph[i][j]>0.6:
            relation_graph[i][j] = 1
            relation_graph[j][i] = 1
        else:
            relation_graph[i][j] = 0
            relation_graph[j][i] = 0

# model information
#r1 --- node 16
#unrelated fault [2,3,4]
uf1 = [2,3,4]
for i in uf1:
    relation_graph[i][16] = 0
    relation_graph[16][i] = 0
#r2 --- node 17
uf2 = [0,1,3,4,5]
for i in uf2:
    relation_graph[i][17] = 0
    relation_graph[17][i] = 0
#r3 --- node 18
uf3 = [0,1,2,3,5]
for i in uf3:
    relation_graph[i][18] = 0
    relation_graph[18][i] = 0

#undirected to directed
#Warning!!!
#pandas DataFrame will choose the cloum firstly and then choose rows.
for i in range(19):
    for j in range(i):
        #debug
        relation_graph[j][i] = 0

relation_graph_np = np.array(relation_graph)
np.save("relation_graph.npy", relation_graph_np)

# np.savetxt("relation_graph.txt", relation_graph)
print(relation_graph)

#save to graphviz
labels = ["F[0:6]",\
          "fe0", "fe1", "fe2", "fe3", "fe4", "fe5", "fe6", "fe7", "fe8", "fe9",\
          "r1", "r2", "r3"]
G = Digraph()
#add nodes
for i in labels:
    G.node(i, i)
#add edges
#fault to all
for i in range(13):
    G.edge(labels[0], labels[i+1])
#feature/res to feature/res
for i in range(6, 19):
    for j in range(i+1, 19):
        if relation_graph_np[i, j] == 1:
            G.edge(labels[i-5], labels[j-5])
print(G)
