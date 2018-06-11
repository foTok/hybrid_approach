"""
analyze the relationship
fault   -- feature    //priori
fault   -- residual   //priori
feature -- feature    //learned
feature -- residual   //learned
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
from ddd.utilities import organise_data
from graph_model.utilities import priori_knowledge
from graph_model.utilities import graphviz_Bayes

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
PGM_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl"
struct_file_gv = "linear_structure.gv"
struct_file_np = "linear_graph.npy"
step_len=100
batch = 20000
fea_num = 12
n = 6 + fea_num + 3

#load fe
FE = torch.load(ANN_PATH+fe_file)
FE.eval()

#priori knowledge
graph = priori_knowledge()

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)


inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)
df = pd.DataFrame(batch_data)
# pearson
relation_graph = df.corr()
relation_graph = abs(relation_graph)
# no edges between features with correfficency < 0.6
for i in range(6, n):
    for j in range(i+1, n):
        if relation_graph[i][j]>0.6 and graph[i, j] == 0:
            graph[i, j] = 1

for i in range(n):
    for j in range(n):
        if graph[i,j]==-1:
            graph[i,j] = 0

np.save(PGM_PATH + struct_file_np, graph)
graphviz_Bayes(graph, PGM_PATH + struct_file_gv ,fea_num)
