"""
This file diagnosis BPSK system by MBD
"""

import os
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
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
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
#[  9.33880161e-01   6.61198389e-02   5.00000000e-04   9.99500000e-01]
print('Finished Training')
