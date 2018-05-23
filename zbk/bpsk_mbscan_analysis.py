"""
analyze mbscan for fault isolation
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.bpsk_block_scan_diagnoser import DiagnoerBlockScan
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ann_diagnoser.loss_function import CrossEntropy
from ann_diagnoser.loss_function import MSE
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np


def hist_eva(label, pro):
    """
    analyze pro
    label: numpy array, batch × fault
    pro:   numpy array, batch × fault
    """
    batch = len(label)
    #eva: batch × fault × hist
    eva = np.array([[[0]*10]*6]*batch)
    for i in range(batch):
        predict = pro[i]
        r_label = label[i]
        p_label = predict > 0.5
        hist = (predict - 1e-6)*10
        hist = hist.astype(np.int32)
        succ = (p_label == r_label)
        for j, k, s in zip(range(6), hist, succ):
            eva[i, j, k] = 1 if s else 2
    #eva_p: fault × hist
    eva_p = []
    for i in range(6):
        hist_i = eva[:, i, :]
        all_p = (np.sum(hist_i!=0, axis=0)+1)
        succ_p = np.sum(hist_i==1, axis=0)
        hist_i_p =  succ_p / all_p
        eva_p.append(hist_i_p)
    eva_p = np.array(eva_p)
    return eva_p

#bpsk_mbs_isolator_para3
#prepare data
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
MODEL_PATH = PATH + "\\ann_model\\"
model_name = "bpsk_mbs_isolator0.pkl"
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

isolator = torch.load(MODEL_PATH + model_name)
isolator.eval()
batch = 1000
epoch = 1000
eva = np.array([[0]*10]*6)
for i in range(epoch):
    inputs, labels, _, _ = mana.random_batch(batch, normal=0, single_fault=10, two_fault=1)
    outputs = isolator(inputs)
    outputs = outputs.data.numpy()
    labels = labels.data.numpy()
    hist_i = hist_eva(labels, outputs)
    eva = eva + hist_i/epoch

print(eva)
