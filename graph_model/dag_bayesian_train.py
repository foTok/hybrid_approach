"""
train a DAG Bayesian network
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

#priori knowledge
#0 ~can not determine
#1 ~connection
#-1~no connection
#initially, all connection can not be determined.
pri_knowledge = np.zeros((19, 19))
#no self connection and downstair connection
for i in range(19):
    for j in range(i+1):
        pri_knowledge[i,j] = -1
#no connection between faults
for i in range(6):
    for j in range(i+1, 6):
        pri_knowledge[i, j] = -1
        pri_knowledge[j, i] = -1
#connections from faults to features or residuals
for i in range(6):
    for j in range(6,19):
        pri_knowledge[i, j] = 1
# model information
#r1 --- node 16
#unrelated fault [2,3,4]
uf1 = [2,3,4]
for i in uf1:
    pri_knowledge[i][16] = -1
    pri_knowledge[16][i] = -1
#r2 --- node 17
uf2 = [0,1,3,4,5]
for i in uf2:
    pri_knowledge[i][17] = -1
    pri_knowledge[17][i] = -1
#r3 --- node 18
uf3 = [0,1,2,3,5]
for i in uf3:
    pri_knowledge[i][18] = -1
    pri_knowledge[18][i] = -1
#no connection between residuals
for i in range(16, 19):
    for j in range(16, 19):
        pri_knowledge[i, j] = -1

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
PGM_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl"
struct_file = "Greedy_Bayes.gv"
pgm_file = "Greedy_Bayes.bn"
step_len=100
epoch = 200
batch = 2000

#load fe
FE = torch.load(ANN_PATH+fe_file)
FE.eval()

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

#labels, 6; features, 10 (12-10); residuals 3.
BL = Bayesian_learning(6+10+3)
BL.set_priori(pri_knowledge)
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=10, two_fault=0)
    feature = FE.fe(inputs)
    batch_data = organise_data(inputs, labels, res, feature)

    # #for test
    # hist_batch(batch_data)

    # #for test
    # scatter_batch(batch_data)

    BL.set_batch(batch_data)
    BL.step(i)

best, _ = BL.best_candidate()
graphviz_Bayes(best.struct, PGM_PATH + struct_file)

BN = BL.best_BN()
s = pickle.dumps(BN)
with open(PGM_PATH + pgm_file, "wb") as f:
    f.write(s)
