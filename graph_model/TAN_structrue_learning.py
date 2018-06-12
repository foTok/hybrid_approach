"""
Learning the TAN classifiers for 6 faults independently and a merged TAN classifier
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import pickle
import numpy as np
import matplotlib.pyplot as pl
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.min_span_tree import mst_learning
from graph_model.utilities import priori_knowledge
from graph_model.utilities import graphviz_Bayes
from graph_model.utilities import compact_graphviz_struct
from ddd.utilities import organise_data

#data amount
small_data = False

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
PGM_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl" if not small_data else "FE1.pkl"
file_flag = "_0" if not small_data else "_1"
step_len=100
batch = 2000 if not small_data else 200

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)
#load fe
FE = torch.load(ANN_PATH+fe_file)
FE.eval()
#sample data
inputs, labels, _, res = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)

#the traits
traits = batch_data[:, 6:]

#learning minimal span tree
mst_l = mst_learning()
mst_l.set_batch(traits)
mst = mst_l.learn_mst()

#feature number
_, fea_num=traits.shape
fea_num = fea_num -3
#priori knowledge
pri_knowledge = priori_knowledge(fea_num)
n = len(pri_knowledge)
#labels
fault_labels = []
for i in range(6):
    fault_labels.append(("f"+str(i), "yellow"))
fea_labels = []
for i in range(fea_num):
    fea_labels.append(("fe"+str(i), "green"))
res_labels = []
for i in range(3):
    res_labels.append(("r"+str(i), "red"))
labels = fault_labels + fea_labels + res_labels

#for TAN
for i in range(6):
    struct = np.zeros((n, n))
    #priori
    kids = pri_knowledge[i, :]
    for j in range(n):
        if kids[j]==1:
            struct[i, j]=1
    #add mst
    t_num = len(mst)
    for j0 in range(t_num):
        for j1 in range(t_num):
            if mst[j0, j1]==1:
                struct[j0+6, j1+6] = 1
    #save it
    file_name = "tan_"+str(i)+file_flag
    compact_graphviz_struct(struct, PGM_PATH+file_name + ".gv", labels)
    np.save(PGM_PATH + file_name, struct)
#for MTAN
struct = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if pri_knowledge[i, j]==1:
            struct[i, j]=1
        if i>=6 and j>=6:
            if mst[i-6, j-6] == 1:
                struct[i, j]=1
graphviz_Bayes(struct, PGM_PATH + "mtan.gv", fea_num)
print("Done")
