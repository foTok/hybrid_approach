"""
hybrid isolator
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import pickle
import time
import numpy as np
import matplotlib.pyplot as pl
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.Bayesian_learning import Bayesian_structure
from graph_model.Bayesian_learning import Bayesian_learning
from ddd.utilities import organise_data
from ddd.utilities import organise_tensor_data
from hybrid_algorithm.hybrid_annbn_diagnoser import hybrid_annbn_diagnoser
from hybrid_algorithm.hybrid_ann_diagnoser import hybrid_ann_diagnoser
from hybrid_algorithm.utilities import priori_vec2tup
from hybrid_algorithm.hybrid_stats import hybrid_stats
from hybrid_algorithm.hybrid_tan_diagnoser import hybrid_tan_diagnoser
from hybrid_algorithm.hybrid_ann1svm_diagnoser import ann1svm_diagnoser
from graph_model.utilities import priori_knowledge

#data amount
small_data = True
#settings
PATH            = parentdir
DATA_PATH       = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH        = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
SVM_PATH        = PATH + "\\ddd\\svm_model\\" + ("big_data\\" if not small_data else "small_data\\")
GRAPH_PATH      = PATH + "\\graph_model\\pg_model\\" + ("big_data\\" if not small_data else "small_data\\")
fe_file         = "FE.pkl"
dia_file        = "DIA.pkl"
svm_file        = "likelihood.m"
hdia_file       = "HDIA.pkl"
mtan_file       = "MTAN.bn"
gsan_file       = "GSAN.bn"
tan_file_prefix = "TAN"
step_len        = 100
batch           = 1000

#load fe and iso
FE = torch.load(ANN_PATH + fe_file)
FE.eval()
DIA = torch.load(ANN_PATH + dia_file)
DIA.eval()
HDIA = torch.load(ANN_PATH + hdia_file)
HDIA.eval()

#load gsan model
with open(GRAPH_PATH + gsan_file, "rb") as f:
    gsan = f.read()
    gsan_model = pickle.loads(gsan)

#load gsan model
with open(GRAPH_PATH + mtan_file, "rb") as f:
    mtan = f.read()
    mtan_model = pickle.loads(mtan)

#priori knowledge
pri_knowledge = priori_knowledge()

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

inputs, labels, _, res = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
#priori by data
ann_start = time.clock()
priori_by_data = DIA(inputs).detach().numpy()
ann_cost  = time.clock() - ann_start
#priori by hybrid
sen_res = organise_tensor_data(inputs, res)
hann_start = time.clock()
priori_by_hybrid = HDIA(sen_res).detach().numpy()
hann_cost = time.clock() - hann_start
#feature
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)
labels = labels.detach().numpy()

#stats
statistic = hybrid_stats()

#hybrid diagnosers
#ANN
ann = hybrid_ann_diagnoser()
#ANN + 1SVM
ann1svm = ann1svm_diagnoser(0.99)
ann1svm.load_svm(SVM_PATH + svm_file)

#Hybrid ANN
hann = hybrid_ann_diagnoser()
#TAN
anntan = hybrid_tan_diagnoser()
anntan.load_file(GRAPH_PATH + tan_file_prefix)
anntan.set_priori_knowledge(pri_knowledge)
#ANN + MTAN
annmtan = hybrid_annbn_diagnoser()
annmtan.set_graph_model(mtan_model)
#ANN + GSAN
anngsan = hybrid_annbn_diagnoser()
anngsan.set_graph_model(gsan_model)

#set order
order = (0,1,2,3,4,5)
ann.set_order(order)
ann1svm.set_order(order)
hann.set_order(order)
anntan.set_order(order)
annmtan.set_order(order)
anngsan.set_order(order)

statistic.add_diagnoser("ann")
statistic.add_diagnoser("ann1svm")
statistic.add_diagnoser("hann")
statistic.add_diagnoser("anntan")
statistic.add_diagnoser("annmtan")
statistic.add_diagnoser("anngsan")

#diagnosis number
num = 1
for label, d_priori, h_priori, data, index in zip(labels, priori_by_data, priori_by_hybrid, batch_data, range(len(labels))):
    print("sample ", index)
    priori_d = priori_vec2tup(d_priori)
    priori_h = priori_vec2tup(h_priori)
    obs = []
    for i in range(6, len(data)):
        obs.append((i, data[i]))
    traits = data[6:]

    #set priori probability
    ann.set_priori(priori_d)
    ann1svm.set_priori(priori_d)
    hann.set_priori(priori_h)
    anntan.set_priori(priori_d)
    annmtan.set_priori(priori_d)
    anngsan.set_priori(priori_d)

    #add obs
    ann1svm.add_obs(traits)
    anntan.add_obs(obs)
    annmtan.add_obs(obs)
    anngsan.add_obs(obs)

    #update
    anntan.update_priori()

    dia_ann      = ann.search(num)
    dia_ann1svm  = ann1svm.search(num)
    dia_hann     = hann.search(num)
    dia_anntan   = anntan.search(num)
    dia_annmtan  = annmtan.search(num)
    dia_anngsan  = anngsan.search(num)

    statistic.append_label(label)
    statistic.append_predicted("ann",       dia_ann)
    statistic.append_predicted("ann1svm",   dia_ann1svm)
    statistic.append_predicted("hann",      dia_hann)
    statistic.append_predicted("anntan",    dia_anntan)
    statistic.append_predicted("annmtan",   dia_annmtan)
    statistic.append_predicted("anngsan",   dia_anngsan)

statistic.print_stats()
#ann time cost
print("ann cost=",           ann_cost)
print("hann cost=",          hann_cost)
#ann time search cost
print("ann time cost=",      ann.time_cost())
print("ann1svm0 time cost=", ann1svm.time_cost())
print("hann time cost=",     hann.time_cost())
print("anntan time cost=",   anntan.time_cost())
print("annmtan time cost=",  annmtan.time_cost())
print("anngsan time cost=",  anngsan.time_cost())

print("DONE")
