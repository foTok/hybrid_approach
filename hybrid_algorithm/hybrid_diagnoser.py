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
from graph_model import graph_component as graph_component
from ddd.utilities import organise_data
from hybrid_algorithm.hybrid_search import hybrid_search
from hybrid_algorithm.utilities import priori_vec2tup

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
GRAPH_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl"
iso_file = "DIA0.pkl"
graph_file = "Greedy_Bayes.bn"
step_len=100
batch = 1000

#load fe and iso
FE = torch.load(ANN_PATH + fe_file)
FE.eval()
ISO = torch.load(ANN_PATH + iso_file)
ISO.eval()

#load graph model
with open(GRAPH_PATH + graph_file, "rb") as f:
    graph = f.read()
    graph_model = pickle.loads(graph)

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)
priori_probability = ISO(inputs).detach().numpy()
label = labels.detach().numpy()

all_ddd = np.round(priori_probability)
all_hybrid = []

for label0, priori0, data in zip(label, priori_probability, batch_data):
    #get priori
    priori = priori_vec2tup(priori0)
    hs = hybrid_search()
    #load graph model
    hs.set_graph_model(graph_model)
    #set order
    hs.set_order((0,1,2,3,4,5))
    #set priori probability
    hs.set_priori(priori)
    #add obs
    for i in range(6, len(data)):
        hs.add_obs(i, data[i])
    r = hs.search()

    #store results
    all_hybrid.append(r[0][1])
    print(label0, np.round(priori0), r[0][1])
all_hybrid = np.array(all_hybrid)

#ddd
d_count = 0
d_a = (label == all_ddd)
for i in d_a:
    d_count = d_count + (1 if i.all() else 0)
print(d_count)

h_count = 0
h_a = (label == all_hybrid)
for i in h_a:
    h_count = h_count + (1 if i.all() else 0)
print(h_count)

print("DONE")
