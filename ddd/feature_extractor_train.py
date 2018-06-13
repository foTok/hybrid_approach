"""
train the feature extractor
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
from ann_diagnoser.bpsk_block_scan_feature_extracter import BlockScanFE
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ann_diagnoser.loss_function import CrossEntropy

#data amount
small_data = False
#prepare data
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
step_len=100
criterion = CrossEntropy
fe_name = "FE0.pkl" if not small_data else "FE1.pkl"

mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

FE = BlockScanFE()
optimizer = optim.Adam(FE.parameters(), lr=0.001, weight_decay=8e-3)
print(FE)

#train
epoch = 2000
batch = 2000 if not small_data else 200
train_loss = []
running_loss = 0.0
if small_data:
    inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
for i in range(epoch):
    if not small_data:
        inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
    optimizer.zero_grad()
    outputs = FE(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    loss_i = loss.item()
    running_loss += loss_i
    train_loss.append(loss_i)
    if i % 10 == 9:
        print('%d loss: %.5f' %(i + 1, running_loss / 10))
        running_loss = 0.0
print('Finished Training')
#save model
torch.save(FE, ANN_PATH + fe_name)

#figure
pl.figure(1)
pl.plot(np.array(train_loss))
pl.title("Training Loss")
pl.xlabel("Epoch")
pl.ylabel("Loss")
pl.show()
