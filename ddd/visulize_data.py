"""
train a DAG Bayesian network
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import numpy as np
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ddd.utilities import organise_data
from ddd.utilities import hist_batch
from ddd.utilities import scatter_batch

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
PGM_PATH = PATH + "\\graph_model\\pg_model\\"
fe_file = "FE0.pkl"
step_len=100
batch = 2000

#load fe
FE = torch.load(ANN_PATH+fe_file)
FE.eval()

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)


inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
batch_data = organise_data(inputs, labels, res, feature)

# hist_batch(batch_data)
scatter_batch(batch_data)

print("DONE")
