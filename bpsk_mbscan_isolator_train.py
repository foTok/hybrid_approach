"""
the main file to conduct the computation
"""
import os
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


#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
TEST_DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

#set ann fullconnect diagnoser
    #ANN
diagnoser = DiagnoerBlockScan(step_len=mana.step_len())
print(diagnoser)
    #loss function
#criterion = MSE
criterion = CrossEntropy
    #optimizer
#optimizer = optim.Adam(diagnoser.parameters(), lr=0.05, weight_decay=1e-5)
optimizer = optim.SGD(diagnoser.parameters(), lr=0.06, momentum=0.9, weight_decay=1e-3)

    #train
epoch = 2000
batch = 2000

train_loss = []
running_loss = 0.0
for epoch in range(epoch):
    inputs, labels, _ = mana.random_batch(batch, normal=0, single_fault=10, two_fault=4)
    optimizer.zero_grad()
    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.data[0]
    train_loss.append(loss.data[0])
    if epoch % 10 == 9:
        print('%d loss: %.5f' %(epoch + 1, running_loss / 10))
        running_loss = 0.0

print('Finished Training')

#save model
torch.save(diagnoser, "ann_model\\bpsk_mbs_isolator3.pkl")
torch.save(diagnoser.state_dict(), "ann_model\\bpsk_mbs_isolator_para3.pkl")

#choose figure 1
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
diagnoser.eval()
eval_loss = []
batch2 = 1000
epoch2 = 1000
for i in range(epoch2):
    inputs, labels, _ = mana2.random_batch(batch2, normal=0, single_fault=10, two_fault=4)
    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    eval_loss.append(loss.data[0])

#choose figure 2
pl.figure(2)
pl.plot(np.array(eval_loss))
pl.title("Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("Loss")
pl.show()