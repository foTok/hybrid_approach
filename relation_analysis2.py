"""
analyze feature relationship
"""
import os
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
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
FE_PATH = PATH + "\\ann_model\\"
fe_file = "FE0.pkl"
step_len=100
epoch = 2000
batch = 2000

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

# no edges between features with correfficency < 0.5
for i in range(6, 19):
    for j in range(i+1, 19):
        if relation_graph[i][j]>0.6:
            relation_graph[i][j] = 1
            relation_graph[j][i] = 1
        else:
            relation_graph[i][j] = 0
            relation_graph[j][i] = 0

# np.savetxt("relation_graph.txt", relation_graph)
# print(relation_graph)



index = ['F',\
         'fe0', 'fe1', 'fe2', 'fe3', 'fe4', 'fe5', 'fe6', 'fe7', 'fe8', 'fe9',\
         'r1', 'r2', 'r3']

dot = Digraph()
for node in index:
    dot.node(node, node)
for i in range(19):
    for j in range(i+1,19):
        if relation_graph[i][j] == 1:
            dot.edge(index[i], index[j])
print(dot.source)
dot.render('Bayesian0.gv')
