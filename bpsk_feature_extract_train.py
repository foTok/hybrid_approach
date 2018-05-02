"""
train the feature extractor
"""
import os
from ann_diagnoser.bpsk_block_scan_feature_extracter import BlockScanFE
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
from tensorboardX import SummaryWriter

#visual
writer = SummaryWriter()

#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
step_len=100
criterion = CrossEntropy

mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

FE = BlockScanFE()
optimizer = optim.Adam(FE.parameters(), lr=0.001, weight_decay=8e-3)
print(FE)

#train
epoch = 3000
batch = 2000
train_loss = []
running_loss = 0.0
#add graph flag
agf = False
for i in range(epoch):
    inputs, labels, _, _ = mana.random_batch(batch, normal=0, single_fault=10, two_fault=1)
    if not agf:                         #visual
        writer.add_graph(FE, inputs)    #visual
        agf = True                      #visual
    optimizer.zero_grad()
    outputs = FE(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    loss_i = loss.item()
    writer.add_scalar('Loss', loss_i, i) #visual
    running_loss += loss_i
    train_loss.append(loss_i)
    if i % 10 == 9:
        print('%d loss: %.5f' %(i + 1, running_loss / 10))
        running_loss = 0.0
print('Finished Training')

#save model
torch.save(FE, "ann_model\\FE1.pkl")

#visual
writer.close()

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
FE_test = torch.load("ann_model\\FE1.pkl")
FE_test.eval()
eval_loss = []
batch2 = 1000
epoch2 = 1000
for i in range(epoch2):
    inputs, labels, _, _ = mana2.random_batch(batch2, normal=0, single_fault=10, two_fault=1)
    outputs = FE_test(inputs)
    loss = criterion(outputs, labels)
    eval_loss.append(loss.item())

#figure 2
pl.figure(2)
pl.plot(np.array(eval_loss))
pl.title("Random Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("Loss")
pl.show()
