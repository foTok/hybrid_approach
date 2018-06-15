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
obj = "fe" #fe, dia, hdia
PATH = parentdir
TEST_DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
step_len=100
criterion = CrossEntropy
norm = False
if obj == "fe":
    dia_name = "FE.pkl"
elif obj == "dia":
    dia_name = "DIA.pkl"
elif obj == "hdia":
    dia_name = "HDIA.pkl"
    norm = True
else:
    print("unkown object!")
    exit(0)

mana = BpskDataTank()
list_files = get_file_list(TEST_DATA_PATH)
for file in list_files:
    mana.read_data(TEST_DATA_PATH+file, step_len=step_len, snr=20, norm=norm)
diagnoser = torch.load(ANN_PATH + dia_name)
diagnoser.eval()
eval_loss = []
batch = 1000
epoch = 100
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0.2, single_fault=10, two_fault=0)
    sen_res = organise_tensor_data(inputs, res)
    outputs = diagnoser(sen_res)
    loss = criterion(outputs, labels)
    eval_loss.append(loss.item())

print("mean loss = ", np.mean(eval_loss))
pl.figure(1)
pl.plot(np.array(eval_loss))
pl.title("Random Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("Loss")
pl.show()
