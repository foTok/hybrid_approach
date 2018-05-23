"""
graph model based isolator
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
from graph_model.utilities import organise_data
from graph_model.utilities import hist_batch
from graph_model.utilities import scatter_batch
from graph_model.utilities import graphviz_Bayes
from hybrid_algorithm.hybrid_search import hybrid_search
from graph_model import graph_component as graph_component

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
FE_PATH = PATH + "\\ddd\\ann_model\\"
GRAPH_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl"
graph_file = "Greedy_Bayes.bn"
step_len=100
batch = 100
priori = ((0.99, 0.01),(0.99, 0.01),(0.99, 0.01),(0.99, 0.01),(0.99, 0.01),(0.99, 0.01))

#load fe
FE = torch.load(FE_PATH+fe_file)
FE.eval()
#load graph model
with open(GRAPH_PATH + graph_file, "rb") as f:
    graph = f.read()
    graph_model = pickle.loads(graph)

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)


inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=10, two_fault=5)
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)
label = labels.detach().numpy()
all_mbd = []

for label0, data in zip(label, batch_data):
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
    all_mbd.append(r[0][1])
    print(label0, r[0][1])

all_mbd = np.array(all_mbd)
m_count = 0
m_a = (label == all_mbd)
for i in m_a:
    m_count = m_count + (1 if i.all() else 0)
print(m_count)
print("DONE")
