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
from hybrid_algorithm.hybrid_ann1svm_diagnoser import ann1svm_diagnoser
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

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=0, two_fault=10)
#priori by data
priori_by_data = DIA(inputs).detach().numpy()
#feature
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)
labels = labels.detach().numpy()

#stats
statistic = hybrid_stats()

#hybrid diagnosers
ann1svm = ann1svm_diagnoser()
#set order
order = (0,1,2,3,4,5)
ann1svm.set_order(order)

statistic.add_diagnoser("ann1svm")

#diagnosis number
num = 1
for label, d_priori, data, index in zip(labels, priori_by_data, batch_data, range(len(labels))):
    print("sample ", index)
    priori_d = priori_vec2tup(d_priori)
    obs = data[6:]
    
    #set priori probability
    ann1svm.set_priori(priori_d)

    #add obs
    ann1svm.add_obs(obs)

    dia_ann1svm     = ann1svm.search(num)

    statistic.append_label(label)
    statistic.append_predicted("ann", dia_ann1svm)


statistic.print_stats()
print("DONE")
