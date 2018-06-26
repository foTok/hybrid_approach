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
from data_manger.utilities import get_file_list
from ann_diagnoser.loss_function import CrossEntropy
from ann_diagnoser.bpsk_higsecnn_diagnoser import higsecnn_diagnoser
from ddd.utilities import organise_tensor_data

#settings
snr             = 20
TIME            = 0.0001
FAULT_TIME      = TIME / 2
N               = 3
fault_type      = ["tma", "tmb", "pseudo_rate"]
rang            = [[0.2, (0.8 * 10**6, 7.3 * 10**6), -0.05], [0.9, (8.8 * 10**6, 13 * 10**6), 0.05]]
pref            = 3 #1->single-fault, 2->two-fault, 3->single-,two-fault
loss            = 0.05
grids           = [0.1, 1.41421*10**6, 0.01]
diagnoser       = higsecnn_diagnoser()
lr              = 1e-3
weight_decay    = 8e-3
training_data   = BpskDataTank()
para_set        = {}
data_path       = parentdir + "\\bpsk_navigate\\data\\small_data\\"
generated_path  = parentdir + "\\ddd\\data\\" + str(snr) + "db\\"
ann_path        = parentdir + "\\ddd\\ann_model\\small_data\\"  + str(snr) + "db\\"
mana0           = BpskDataTank()                       #historical data
mana1           = BpskDataTank()                       #generated data
step_len        = 100
batch           = 1000
criterion       = CrossEntropy
epoch1          = 500
epoch0          = 1000
file_name       = "higsecnn.pkl"

#embedded part only
optimizer1      = optim.Adam(diagnoser.embedded_parameters(), lr=lr, weight_decay=weight_decay)
while True:
    parameters = sample_parameters(N, fault_type, grids, rang[0], rang[1], pref, para_set)
    file_list  = sample_data(FAULT_TIME, TIME, parameters, generated_path)
    print("file_list=",file_list)
    if len(file_list) == 0:
        break
    #a tmp data manager
    mana_t = BpskDataTank()
    add_files(generated_path, file_list, mana_t, step_len, snr=snr)
    #sample batch
    _, labels, _, res = mana_t.random_batch(batch, normal=0.25, single_fault=10, two_fault=1)
    r0      = np.array([r[0] for r in res])
    r0      = torch.Tensor(r0)
    labels1 = labels[:, [0, 1, 5]]
    output1 = diagnoser.embedded_forward(r0)
    loss1   = criterion(output1, labels1)
    print("evaluation loss=", loss1.item())
    if loss1 > loss:
        add_files(generated_path, file_list, mana1, step_len, snr=snr)
        while loss1.item() > loss:
            _, labels, _, res = mana1.random_batch(batch, normal=0.25, single_fault=10, two_fault=1)
            r0      = np.array([r[0] for r in res])
            r0      = torch.Tensor(r0)
            labels1  = labels[:, [0, 1, 5]]
            optimizer1.zero_grad()
            output1 = diagnoser.embedded_forward(r0)
            loss1   = criterion(output1, labels1)
            loss1.backward()
            optimizer1.step()
            print("generated training loss=", loss1.item())
    else:
        break
#continue optimazation
for i in range(epoch1):
    _, labels, _, res = mana1.random_batch(batch, normal=0.25, single_fault=10, two_fault=1)
    r0      = np.array([r[0] for r in res])
    r0      = torch.Tensor(r0)
    labels1  = labels[:, [0, 1, 5]]
    optimizer1.zero_grad()
    output1 = diagnoser.embedded_forward(r0)
    loss1   = criterion(output1, labels1)
    loss1.backward()
    optimizer1.step()
    print("generated training loss ", i, "=", loss1.item())
#historical data training
list_files     = get_file_list(data_path)
for file in list_files:
    mana0.read_data(data_path+file, step_len=step_len, snr=snr, norm=True)
diagnoser.freeze_share()
optimizer0     = optim.Adam(diagnoser.parameters(), lr=lr, weight_decay=weight_decay)
for i in range(epoch0):
    #historical data
    inputs0, labels0, _, res0 = mana0.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
    sen_res = organise_tensor_data(inputs0, res0)
    #optimization
    optimizer0.zero_grad()
    outputs0 = diagnoser(sen_res)
    loss0    = criterion(outputs0, labels0)
    loss     = loss0
    loss.backward()
    optimizer0.step()
    print("historical training loss", i, "=", loss.item())
diagnoser.thaw_share()
#save model
torch.save(diagnoser, ann_path + file_name)
