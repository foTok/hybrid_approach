"""
This file finding probability of residuals for fault isolation
"""

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import torch
import matplotlib.pyplot as pl
from numpy.random import rand
from bpsk_navigate.bpsk_generator import Bpsk
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.bpsk_data_tank import parse_filename
from data_manger.utilities import get_file_list
from data_manger.utilities import statistic
from mbd.res_Z_test import Z_test

#prepare data
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)
mana.info()


#train
epoch = 50
batch = 1000

normal_mean = []
fault_mean = []
normal_var = []
fault_var = []
for i in range(epoch):
    _, _, _, normal_res = mana.random_batch(batch, normal=1,single_fault=1, two_fault=1)
    normal_res = np.array(normal_res)
    normal_res = np.abs(normal_res)
    n_mean = np.mean(normal_res, axis=2)
    n_var = np.var(n_mean, axis=0)
    normal_var.append(n_var)
    n_mean = np.mean(n_mean, axis=0)
    normal_mean.append(n_mean)
    _, _, _, fault_res = mana.random_batch(batch, normal=1,single_fault=1, two_fault=1)
    fault_res = np.array(fault_res)
    fault_res = np.abs(fault_res)
    f_mean = np.mean(fault_res, axis=2)
    f_var = np.var(f_mean, axis=0)
    fault_var.append(f_var)
    f_mean = np.mean(f_mean, axis=0)
    fault_mean.append(f_mean)

normal_mean = np.array(normal_mean)
normal_mean = np.mean(normal_mean, axis=0)

fault_mean = np.array(fault_mean)
fault_mean = np.mean(fault_mean, axis=0)

normal_var = np.array(normal_var)
normal_var = np.mean(normal_var, axis=0)

fault_var = np.array(fault_var)
fault_var = np.mean(fault_var, axis=0)

print("normal_mean={}".format(normal_mean))
print("fault_mean={}".format(fault_mean))
print("normal_var={}".format(normal_var))
print("fault_var={}".format(fault_var))
# normal_mean=[ 0.08004993  0.05648239  0.05651077  0.61044511]
# fault_mean=[ 0.08004375  0.05648811  0.0565525   0.61049365]
# normal_var=[  3.61122403e-05   1.90020324e-05   2.22642168e-05   8.92350168e-03]
# fault_var=[  3.61650984e-05   1.92340598e-05   2.21170949e-05   8.92519620e-03]
