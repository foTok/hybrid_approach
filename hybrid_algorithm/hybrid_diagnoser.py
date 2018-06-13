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
from ddd.utilities import organise_data
from ddd.utilities import organise_tensor_data
from hybrid_algorithm.hybrid_annbn_diagnoser import hybrid_annbn_diagnoser
from hybrid_algorithm.hybrid_ann_diagnoser import hybrid_ann_diagnoser
from hybrid_algorithm.utilities import priori_vec2tup
from hybrid_algorithm.hybrid_stats import hybrid_stats
from hybrid_algorithm.hybrid_tan_diagnoser import hybrid_tan_diagnoser
from graph_model.utilities import priori_knowledge

#data amount
small_data = False
#settings
PATH            = parentdir
DATA_PATH       = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH        = PATH + "\\ddd\\ann_model\\"
GRAPH_PATH      = PATH + "\\graph_model\\pg_model\\"
fe_file         = "FE0.pkl" if not small_data else "FE1.pkl"
dia_file        = "DIA0.pkl" if not small_data else "DIA1.pkl"
hdia_file       = "HDIA0.pkl" if not small_data else "HDIA1.pkl"
mtan_file       = "MTAN0.bn"  if not small_data else "MTAN1.bn"
gsan_file       = "GSAN0.bn" if not small_data else "GSAN1.bn"
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
#ANN
ann = hybrid_ann_diagnoser()
#Hybrid ANN
hann = hybrid_ann_diagnoser()
#TAN
anntan = hybrid_tan_diagnoser()
anntan.load_file(GRAPH_PATH + tan_file_prefix, "0")
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
hann.set_order(order)
anntan.set_order(order)
annmtan.set_order(order)
anngsan.set_order(order)

statistic.add_diagnoser("ann")
statistic.add_diagnoser("hann")
statistic.add_diagnoser("anntan")
statistic.add_diagnoser("annmtan")
statistic.add_diagnoser("anngsan")

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
    anntan.set_priori(priori_d)
    annmtan.set_priori(priori_d)
    anngsan.set_priori(priori_d)

    #add obs
    anntan.add_obs(obs)
    annmtan.add_obs(obs)
    anngsan.add_obs(obs)

    #update
    anntan.update_priori()

    dia_ann     = ann.search(num)
    dia_hann    = hann.search(num)
    dia_anntan     = anntan.search(num)
    dia_annmtan = annmtan.search(num)
    dia_anngsan = anngsan.search(num)

    statistic.append_label(label)
    statistic.append_predicted("ann", dia_ann)
    statistic.append_predicted("hann", dia_hann)
    statistic.append_predicted("anntan", dia_anntan)
    statistic.append_predicted("annmtan", dia_annmtan)
    statistic.append_predicted("anngsan", dia_anngsan)

statistic.print_stats()
print("ann time cost=", ann.time_cost())
print("hann time cost=", hann.time_cost())
print("anntan time cost=", anntan.time_cost())
print("annmtan time cost=", annmtan.time_cost())
print("anngsan time cost=", anngsan.time_cost())

print("DONE")
