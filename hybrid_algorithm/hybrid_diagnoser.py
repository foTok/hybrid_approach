"""
hybrid isolator
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
from graph_model.Bayesian_learning import Bayesian_structure
from graph_model.Bayesian_learning import Bayesian_learning
from graph_model.utilities import graphviz_Bayes
from ddd.utilities import organise_data
from ddd.utilities import organise_tensor_data
from hybrid_algorithm.hybrid_annbn_diagnoser import hybrid_annbn_diagnoser
from hybrid_algorithm.hybrid_ann_diagnoser import hybrid_ann_diagnoser
from hybrid_algorithm.utilities import priori_vec2tup
from hybrid_algorithm.hybrid_stats import hybrid_stats

#data amount
small_data = False
#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
GRAPH_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl" if not small_data else "FE1.pkl"
dia_file = "DIA0.pkl" if not small_data else "DIA1.pkl"
hdia_file = "HDIA0.pkl" if not small_data else "HDIA0.pkl"
graph_file = "GSAN0.bn" if not small_data else "GSAN0.bn"
step_len=100
batch = 1000

#load fe and iso
FE = torch.load(ANN_PATH + fe_file)
FE.eval()
DIA = torch.load(ANN_PATH + dia_file)
DIA.eval()
HDIA = torch.load(ANN_PATH + hdia_file)
HDIA.eval()

#load graph model
with open(GRAPH_PATH + graph_file, "rb") as f:
    graph = f.read()
    graph_model = pickle.loads(graph)

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=0, two_fault=10)
#priori by data
priori_by_data = DIA(inputs).detach().numpy()
#priori by hybrid
sen_res = organise_tensor_data(inputs, res)
priori_by_hybrid = HDIA(sen_res).detach().numpy()
#feature
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)
labels = labels.detach().numpy()

#stats
statistic = hybrid_stats()

#hybrid diagnosers
ann = hybrid_ann_diagnoser()
hann = hybrid_ann_diagnoser()
annbn = hybrid_annbn_diagnoser()
annbn.set_graph_model(graph_model)

#set order
order = (0,1,2,3,4,5)
ann.set_order(order)
hann.set_order(order)
annbn.set_order(order)

statistic.add_diagnoser("ann")
statistic.add_diagnoser("hann")
statistic.add_diagnoser("annbn")

#diagnosis number
num = 3
for label, d_priori, h_priori, data, index in zip(labels, priori_by_data, priori_by_hybrid, batch_data, range(len(labels))):
    print("sample ", index)
    priori_d = priori_vec2tup(d_priori)
    priori_h = priori_vec2tup(h_priori)
    obs = []
    for i in range(6, len(data)):
        obs.append((i, data[i]))

    #set priori probability
    ann.set_priori(priori_d)
    hann.set_priori(priori_h)
    annbn.set_priori(priori_d)
    #add obs
    annbn.add_obs(obs)

    dia_ann   = ann.search(num)
    dia_hann  = hann.search(num)
    dia_annbn = annbn.search(num)

    statistic.append_label(label)
    statistic.append_predicted("ann", dia_ann)
    statistic.append_predicted("hann", dia_hann)
    statistic.append_predicted("annbn", dia_annbn)

statistic.print_stats()
print("ann search time=", ann.search_time())
print("hann search time=", hann.search_time())
print("annbn search time=", annbn.search_time())

print("DONE")
