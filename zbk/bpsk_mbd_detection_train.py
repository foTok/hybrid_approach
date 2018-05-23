"""
This file finding probability of residuals for fault detection
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
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)
mana.info()


#train
epoch = 2000
batch = 2000
cpt = []
for i in range(epoch):
    print("epoch = {}".format(i))
    inputs, labels, _, res = mana.random_batch(batch, normal=0.5,single_fault=10, two_fault=4)
    labels = (torch.sum(labels, 1) > 0).float().view(-1, 1)
    labels = labels.data.numpy()
    Z = Z_test(res, 0.98)
    Z = [1 if True in i else 0 for i in Z]
    p = statistic(labels, Z)
    cpt.append(p)

aver_cpt = np.mean(np.array(cpt), axis=0)
print(aver_cpt)
#[ 0.99819285  0.00180715  0.1041675   0.8958325 ]
print('Finished Training')
