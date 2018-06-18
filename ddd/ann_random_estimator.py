"""
estimate the diagnoser or feature extractor randomly
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
from ann_diagnoser.bpsk_block_scan_hybrid_diagnoser import BlockScanHD
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ann_diagnoser.loss_function import CrossEntropy
from ann_diagnoser.loss_function import MSE
from ddd.utilities import organise_tensor_data

#data amount
small_data = True
#settings
obj = ["dia", "hdia","bsshdia"] #fe, dia, hdia, bsshdia
PATH = parentdir
TEST_DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
step_len=100
criterion = CrossEntropy
norm = False
dia_name = []
for dia in obj:
    if dia == "fe":
        model_name = "FE.pkl"
    elif dia == "dia":
        model_name = "DIA.pkl"
    elif dia == "hdia":
        model_name = "HDIA.pkl"
        norm = True
    elif dia == "bsshdia":
        model_name = "BSSHDIA.pkl"
        norm = True
    else:
        print("unkown object!")
        exit(0)
    dia_name.append(model_name)

mana = BpskDataTank()
list_files = get_file_list(TEST_DATA_PATH)
for file in list_files:
    mana.read_data(TEST_DATA_PATH+file, step_len=step_len, snr=20, norm=norm)
#load diagnoser
diagnoser = []
for name in dia_name:
    d = torch.load(ANN_PATH + name)
    d.eval()
    diagnoser.append(d)
batch       = 1000
epoch       = 100
eval_loss   = [0]*len(dia_name)
accuracy    = [0] *len(dia_name)
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0.0, single_fault=10, two_fault=0)
    sen_res = organise_tensor_data(inputs, res)
    for k, d in zip(range(len(dia_name)), diagnoser):
        outputs = d(sen_res)
        loss = criterion(outputs, labels)
        eval_loss[k] = eval_loss[k] + loss.item() / epoch
        prob = outputs.detach().numpy()
        pred = np.round(prob)
        targ = labels.detach().numpy()
        acc  = [c.all() for c in (pred==targ)]
        length = len(acc)
        accuracy[k] = accuracy[k] + np.sum(acc)/(length * epoch)

print("mean loss = ", eval_loss)
print("accuracy = ",  accuracy)
