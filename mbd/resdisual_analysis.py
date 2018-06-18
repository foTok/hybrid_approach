"""
analyze residuals
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from mbd.utilities import residuals_pca

#data amount
small_data      = True
#settings
PATH            = parentdir
DATA_PATH       = PATH + "\\bpsk_navigate\\data\\" + ("big_data\\" if not small_data else "small_data\\")
step_len        = 100
batch           = 10000

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

inputs, labels, _, res = mana.random_batch(batch, normal=1.0, single_fault=0, two_fault=0)
pca = residuals_pca(inputs, res)
var = np.mean(pca**2, 0)
print("var=", var)
print("DONE")
