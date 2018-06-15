"""
the main file to conduct the computation
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
from ann_diagnoser.bpsk_bss_hybrid_diagnoser import BSSHD
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ann_diagnoser.loss_function import CrossEntropy
from ddd.utilities import organise_tensor_data

#data amount
small_data = True
#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\" + ("big_data\\" if not small_data else "small_data\\")
ANN_PATH = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
step_len=100
criterion = CrossEntropy
hdia_name = "BSSHDIA.pkl"

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

diagnoser = BSSHD()
print(diagnoser)
optimizer = optim.Adam(diagnoser.parameters(), lr=0.001, weight_decay=8e-3)

#train
epoch = 1000
batch = 2000 if not small_data else 1000
train_loss = []
running_loss = 0.0
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
    sen_res = organise_tensor_data(inputs, res)
    optimizer.zero_grad()
    outputs = diagnoser(sen_res)
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
torch.save(diagnoser, ANN_PATH + hdia_name)

#figure
pl.figure(1)
pl.plot(np.array(train_loss))
pl.title("Training Loss")
pl.xlabel("Epoch")
pl.ylabel("Loss")
pl.show()
