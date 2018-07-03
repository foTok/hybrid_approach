"""
analyze variable influence relationship.
the output is a directed graph that show how variables influence each other.
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
from ddd.utilities import accuracy
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
snr  = 20
PATH = parentdir
DATA_PATH = parentdir + "\\bpsk_navigate\\data\\" + ("big_data\\" if not small_data else "small_data\\")
ANN_PATH  = parentdir + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\") + str(snr) + "db\\"
step_len=100
criterion = CrossEntropy

mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=snr)

# nn0 = SimpleDiagnoer()
# nn1 = SimpleDiagnoer()
# nn2 = SimpleDiagnoer()
# nn3 = SimpleDiagnoer()
# nn4 = SimpleDiagnoer()

# opt0 = optim.Adam(nn0.parameters(), lr=0.001, weight_decay=5e-3)
# opt1 = optim.Adam(nn1.parameters(), lr=0.001, weight_decay=5e-3)
# opt2 = optim.Adam(nn2.parameters(), lr=0.001, weight_decay=5e-3)
# opt3 = optim.Adam(nn3.parameters(), lr=0.001, weight_decay=5e-3)
# opt4 = optim.Adam(nn4.parameters(), lr=0.001, weight_decay=5e-3)

# #train
# epoch = 1000
# batch = 2000
# print("start training!")
# for i in range(epoch):
#     inputs, labels, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
#     x0 = inputs[:, [0], :]
#     x1 = inputs[:, [1], :]
#     x2 = inputs[:, [2], :]
#     x3 = inputs[:, [3], :]
#     x4 = inputs[:, [4], :]

#     opt0.zero_grad()
#     out0    = nn0(x0)
#     loss0   = criterion(out0, labels)
#     loss0.backward()
#     opt0.step()

#     opt1.zero_grad()
#     out1    = nn1(x1)
#     loss1   = criterion(out1, labels)
#     loss1.backward()
#     opt1.step()

#     opt2.zero_grad()
#     out2    = nn2(x2)
#     loss2   = criterion(out2, labels)
#     loss2.backward()
#     opt2.step()

#     opt3.zero_grad()
#     out3    = nn3(x3)
#     loss3   = criterion(out3, labels)
#     loss3.backward()
#     opt3.step()

#     opt4.zero_grad()
#     out4    = nn4(x4)
#     loss4   = criterion(out4, labels)
#     loss4.backward()
#     opt4.step()

# print('Finished Training')
# #save model
# torch.save(nn0, ANN_PATH + "nn0.pkl")
# torch.save(nn1, ANN_PATH + "nn1.pkl")
# torch.save(nn2, ANN_PATH + "nn2.pkl")
# torch.save(nn3, ANN_PATH + "nn3.pkl")
# torch.save(nn4, ANN_PATH + "nn4.pkl")


#test
t_nn0 = torch.load(ANN_PATH + "nn0.pkl")
t_nn1 = torch.load(ANN_PATH + "nn1.pkl")
t_nn2 = torch.load(ANN_PATH + "nn2.pkl")
t_nn3 = torch.load(ANN_PATH + "nn3.pkl")
t_nn4 = torch.load(ANN_PATH + "nn4.pkl")
t_nn0.eval()
t_nn1.eval()
t_nn2.eval()
t_nn3.eval()
t_nn4.eval()
batch2 = 1000
epoch2 = 100
eval_acc = np.zeros((5, 6))
print("Start evaluation!")
for i in range(epoch2):
    inputs, labels, _, _ = mana.random_batch(batch2, normal=0, single_fault=10, two_fault=0)
    x0 = inputs[:, [0], :]
    x1 = inputs[:, [1], :]
    x2 = inputs[:, [2], :]
    x3 = inputs[:, [3], :]
    x4 = inputs[:, [4], :]

    out0        = t_nn0(x0)
    eval_acc[0,:] = eval_acc[0,:] + accuracy(out0, labels) / epoch2

    out1        = t_nn1(x1)
    eval_acc[1,:] = eval_acc[1,:] + accuracy(out1, labels) / epoch2

    out2        = t_nn2(x2)
    eval_acc[2,:] = eval_acc[2,:] + accuracy(out2, labels) / epoch2

    out3        = t_nn3(x3)
    eval_acc[3,:] = eval_acc[3,:] + accuracy(out3, labels) / epoch2

    out4        = t_nn4(x4)
    eval_acc[4,:] = eval_acc[4,:] + accuracy(out4, labels) / epoch2

print("eval_acc=\n", eval_acc)
print("End evaluation!")
