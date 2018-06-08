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
from graph_model.min_span_tree_learning import mst_learning
from graph_model.utilities import priori_knowledge
from graph_model.utilities import graphviz_Bayes
from ddd.utilities import organise_data

#priori knowledge
pri_knowledge = priori_knowledge()

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
PGM_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl"
struct_file_body = "tan"            #"Greedy_Bayes.gv"
pgm_file_body = "tan"               #"Greedy_Bayes.bn"
step_len=100
batch = 20000

#load fe
FE = torch.load(ANN_PATH+fe_file)
FE.eval()

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)

traits = batch_data[:, 6:]

#learning minimal span tree
mst_l = mst_learning()
mst_l.set_batch(traits)
mst = mst_l.learn_mst()

print("Done")
