"""
the main file to conduct the computation
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from ann_diagnoser.bpsk_block_scan_diagnoser import PredictorBlockScan
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


#prepare data
step_len=100
PATH = parentdir
normalization = Variable(torch.Tensor([10, 100, 100, 10, 10, 10**-6, 10**-6]))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
MODEL_PATH = PATH + "\\ann_model\\"
mana = BpskDataTank()

list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

diagnoser = PredictorBlockScan()
print(diagnoser)
criterion = MSE
optimizer = optim.Adam(diagnoser.parameters(), lr=0.005, weight_decay=2e-3)

#train
epoch = 2000
batch = 2000
train_loss = []
running_loss = 0.0
for i in range(epoch):
    #label need normalization
    inputs, _, labels, _ = mana.random_batch(batch, normal=0, single_fault=10, two_fault=4)
    labels = labels * normalization
    optimizer.zero_grad()
    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.data[0]
    train_loss.append(loss.data[0])
    if i % 10 == 9:
        print('%d loss: %.5f' %(i + 1, running_loss / 10))
        running_loss = 0.0

print('Finished Training')

#save model
torch.save(diagnoser, MODEL_PATH + "bpsk_mbs_predictor.pkl")

#figure 1
pl.figure(1)
pl.plot(np.array(train_loss))
pl.title("Training Loss")
pl.xlabel("Epoch")
pl.ylabel("Loss")

#test
TEST_DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
mana2 = BpskDataTank()
list_files2 = get_file_list(TEST_DATA_PATH)
for file in list_files2:
    mana2.read_data(TEST_DATA_PATH+file, step_len=step_len, snr=20)
predictor = torch.load(MODEL_PATH + "bpsk_mbs_predictor.pkl")
predictor.eval()
eval_loss = []
batch2 = 1000
epoch2 = 1000
for i in range(epoch2):
    inputs, _, labels, _ = mana2.random_batch(batch2, normal=0, single_fault=10, two_fault=4)
    labels = labels * normalization
    outputs = predictor(inputs)
    loss = criterion(outputs, labels)
    eval_loss.append(loss.data[0])

#figure 2
pl.figure(2)
pl.plot(np.array(eval_loss))
pl.title("Random Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("Loss")
pl.show()