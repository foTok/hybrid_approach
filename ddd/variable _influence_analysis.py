"""
analyze variable influence relationship.
the output is a directed graph that show how variables influence each other.
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
from ann_diagnoser.simple_diagnoser import SimpleDiagnoer
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ann_diagnoser.loss_function import CrossEntropy
from ann_diagnoser.loss_function import VectorCrossEntropy
from ann_diagnoser.loss_function import MSE
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
#data amount
small_data = True
#prepare data
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\" + ("big_data\\" if not small_data else "small_data\\")
step_len=100
criterion = CrossEntropy

mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

nn0 = SimpleDiagnoer()
nn1 = SimpleDiagnoer()
nn2 = SimpleDiagnoer()
nn3 = SimpleDiagnoer()
nn4 = SimpleDiagnoer()

opt0 = optim.Adam(nn0.parameters(), lr=0.001, weight_decay=5e-3)
opt1 = optim.Adam(nn1.parameters(), lr=0.001, weight_decay=5e-3)
opt2 = optim.Adam(nn2.parameters(), lr=0.001, weight_decay=5e-3)
opt3 = optim.Adam(nn3.parameters(), lr=0.001, weight_decay=5e-3)
opt4 = optim.Adam(nn4.parameters(), lr=0.001, weight_decay=5e-3)

#train
epoch = 2000
batch = 2000
print("start training!")
for i in range(epoch):
    inputs, labels, _, _ = mana.random_batch(batch, normal=0, single_fault=10, two_fault=0)
    x0 = inputs[:, [0], :]
    x1 = inputs[:, [1], :]
    x2 = inputs[:, [2], :]
    x3 = inputs[:, [3], :]
    x4 = inputs[:, [4], :]

    opt0.zero_grad()
    out0    = nn0(x0)
    loss0   = criterion(out0, labels)
    loss0.backward()
    opt0.step()

    opt1.zero_grad()
    out1    = nn1(x1)
    loss1   = criterion(out1, labels)
    loss1.backward()
    opt1.step()

    opt2.zero_grad()
    out2    = nn2(x2)
    loss2   = criterion(out2, labels)
    loss2.backward()
    opt2.step()

    opt3.zero_grad()
    out3    = nn3(x3)
    loss3   = criterion(out3, labels)
    loss3.backward()
    opt3.step()

    opt4.zero_grad()
    out4    = nn4(x4)
    loss4   = criterion(out4, labels)
    loss4.backward()
    opt4.step()

print('Finished Training')
#save model
torch.save(nn0, "ann_model\\nn0.pkl")
torch.save(nn1, "ann_model\\nn1.pkl")
torch.save(nn2, "ann_model\\nn2.pkl")
torch.save(nn3, "ann_model\\nn3.pkl")
torch.save(nn4, "ann_model\\nn4.pkl")



#test
TEST_DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
mana2 = BpskDataTank()
list_files2 = get_file_list(TEST_DATA_PATH)
for file in list_files2:
    mana2.read_data(TEST_DATA_PATH+file, step_len=step_len, snr=20)
t_nn0 = torch.load("ann_model\\nn0.pkl")
t_nn1 = torch.load("ann_model\\nn1.pkl")
t_nn2 = torch.load("ann_model\\nn2.pkl")
t_nn3 = torch.load("ann_model\\nn3.pkl")
t_nn4 = torch.load("ann_model\\nn4.pkl")
t_nn0.eval()
t_nn1.eval()
t_nn2.eval()
t_nn3.eval()
t_nn4.eval()
batch2 = 1000
epoch2 = 1000
eval_loss = np.zeros((5, 6))
print("Start evaluation!")
for i in range(epoch2):
    alpha = i / (i + 1)
    beta  = 1 / (i + 1)
    inputs, labels, _, _ = mana2.random_batch(batch2, normal=0, single_fault=10, two_fault=0)
    x0 = inputs[:, [0], :]
    x1 = inputs[:, [1], :]
    x2 = inputs[:, [2], :]
    x3 = inputs[:, [3], :]
    x4 = inputs[:, [4], :]

    out0    = t_nn0(x0)
    loss0   = VectorCrossEntropy(out0, labels)
    eval_loss[0, :] = alpha * eval_loss[0, :] + beta * loss0

    out1    = t_nn1(x1)
    loss1   = VectorCrossEntropy(out1, labels)
    eval_loss[1, :] = alpha * eval_loss[1, :] + beta * loss1

    out2    = t_nn2(x2)
    loss2   = VectorCrossEntropy(out2, labels)
    eval_loss[2, :] = alpha * eval_loss[2, :] + beta * loss2

    out3    = t_nn3(x3)
    loss3   = VectorCrossEntropy(out3, labels)
    eval_loss[3, :] = alpha * eval_loss[3, :] + beta * loss3

    out4    = t_nn4(x4)
    loss4   = VectorCrossEntropy(out4, labels)
    eval_loss[4, :] = alpha * eval_loss[4, :] + beta * loss4

print("eval_loss=\n", eval_loss)
print("End evaluation!")
