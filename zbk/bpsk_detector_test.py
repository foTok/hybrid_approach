"""
This file diagnosis BPSK system by MBD
"""

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import torch
import matplotlib.pyplot as pl
from scipy import stats
from numpy.random import rand
from bpsk_navigate.bpsk_generator import Bpsk
from data_manger.bpsk_data_tank import parse_filename
from data_manger.utilities import get_file_list
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import statistic
from mbd.res_Z_test import Z_test
from hybrid_algorithm.hybrid_detector import hybrid_detector


#prepare data
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
MODEL_PATH = PATH + "\\ann_model\\"
file_name = "bpsk_mbs_detector0.pkl"
TIME = 0.0001

mana = BpskDataTank()
step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)
mana.info()
detector = torch.load(MODEL_PATH + file_name)
detector.eval()

epoch = 1000
batch = 1000
cpt_m = []
cpt_d = []
cpt_h = []
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0.8,single_fault=10, two_fault=1)
    labels = (torch.sum(labels, 1) > 0).float().view(-1, 1)
    labels = labels.data.numpy()
    #MBD
    Z = Z_test(res, 0.98)
    Z = [1 if True in i else 0 for i in Z]
    p_m = statistic(labels, Z)
    cpt_m.append(p_m)
    #DDD
    d_p = detector(inputs)
    d_p = d_p.data.numpy()
    d_labels = [1 if i > 0.5 else 0 for i in d_p]
    p_d = statistic(labels, d_labels)
    cpt_d.append(p_d)
    #Hybrid
    h_labels = hybrid_detector(Z, d_p)
    p_h = statistic(labels, h_labels)
    cpt_h.append(p_h)

aver_cpt_m = np.mean(np.array(cpt_m), axis=0)
aver_cpt_d = np.mean(np.array(cpt_d), axis=0)
aver_cpt_h = np.mean(np.array(cpt_h), axis=0)
print("MBD:{}".format(aver_cpt_m))
print("DDD:{}".format(aver_cpt_d))
print("Hybrid:{}".format(aver_cpt_h))
