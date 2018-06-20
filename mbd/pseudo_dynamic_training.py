"""
dynamic training means sampleing data based on different paramemeters dynamicly and training diagnoser
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
from bpsk_navigate.bpsk_generator import Bpsk
from bpsk_navigate.utilities import compose_file_name
from data_manger.bpsk_data_tank import BpskDataTank
from mbd.utilities import sample_parameters
from mbd.utilities import sample_data
from mbd.utilities import add_files
from ann_diagnoser.loss_function import CrossEntropy
from ann_diagnoser.bpsk_bs_pseudo_diagnoser import BlockScanPD

#settings
TIME            = 0.0001
FAULT_TIME      = TIME / 2
N               = 3
fault_type      = ["tma", "tmb", "pseudo_rate"]
rang            = [[0.2, (0.8 * 10**6, 7.3 * 10**6), -0.05], [0.9, (8.8 * 10**6, 13 * 10**6), 0.05]]
pref            = 3 #1->single-fault, 2->two-fault, 3->single-,two-fault
loss            = 0.05
grids           = [0.1, 1.41421*10**6, 0.01]
diagnoser       = BlockScanPD()
optimizer       = optim.Adam(diagnoser.parameters(), lr=0.001, weight_decay=8e-3)
training_data   = BpskDataTank()
para_set        = {}
data_path       = parentdir + "\\mbd\\data\\"
ann_path        = parentdir + "\\mbd\\ann_model\\"
mana            = BpskDataTank()
step_len        = 100
batch           = 1000
criterion       = CrossEntropy
epoch           = 500
pdia_name       = "PDIA.pkl"

while True:
    parameters = sample_parameters(N, fault_type, grids, rang[0], rang[1], pref, para_set)
    file_list  = sample_data(FAULT_TIME, TIME, parameters, data_path)
    print("file_list=",file_list)
    if len(file_list) == 0:
        break
    #a tmp data manager
    mana0 = BpskDataTank()
    add_files(data_path, file_list, mana0, step_len, snr=20)
    #sample batch
    _, labels, _, res = mana0.random_batch(batch, normal=0.25, single_fault=10, two_fault=1)
    r0      = np.array([r[0] for r in res])
    r0      = torch.Tensor(r0)
    r0      = r0.view(-1, 1, 100)
    labels  = labels[:, [0, 1, 5]]
    output0 = diagnoser(r0)
    loss0   = criterion(output0, labels)
    print("evaluation loss=", loss0.item())
    if loss0 > loss:
        add_files(data_path, file_list, mana, step_len, snr=20)
        while loss0.item() > loss:
            _, labels, _, res = mana.random_batch(batch, normal=0.25, single_fault=10, two_fault=1)
            r0      = np.array([r[0] for r in res])
            r0      = torch.Tensor(r0)
            r0      = r0.view(-1, 1, 100)
            labels  = labels[:, [0, 1, 5]]
            optimizer.zero_grad()
            output0 = diagnoser(r0)
            loss0   = criterion(output0, labels)
            loss0.backward()
            optimizer.step()
            print("training loss=", loss0.item())
    else:
        break
for _ in range(epoch):
    _, labels, _, res = mana.random_batch(batch, normal=0.25, single_fault=10, two_fault=1)
    r0      = np.array([r[0] for r in res])
    r0      = torch.Tensor(r0)
    r0      = r0.view(-1, 1, 100)
    labels  = labels[:, [0, 1, 5]]
    optimizer.zero_grad()
    output0 = diagnoser(r0)
    loss0   = criterion(output0, labels)
    loss0.backward()
    optimizer.step()
    print("last training loss=", loss0.item())
#save model
torch.save(diagnoser, ann_path + pdia_name)
