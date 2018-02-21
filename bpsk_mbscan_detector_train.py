"""
the main file to conduct the computation
"""
import os
from ann_diagnoser.bpsk_block_scan_diagnoser import DetectorBlockScan
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
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

#set ann fullconnect diagnoser
    #ANN
diagnoser = DetectorBlockScan(step_len=mana.step_len())
print(diagnoser)
    #loss function
#criterion = MSE
criterion = CrossEntropy
    #optimizer
#optimizer = optim.Adam(diagnoser.parameters(), lr=0.05, weight_decay=1e-5)
optimizer = optim.SGD(diagnoser.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)

    #train
episode = 2000
batch = 2000

train_loss = []
running_loss = 0.0
for epoch in range(episode):
    inputs, labels, _ = mana.random_batch(batch, normal=0.5)
    labels = (torch.sum(labels, 1) > 0).float().view(batch, 1)
    optimizer.zero_grad()
    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss.append(loss.data[0])
    running_loss += loss.data[0]
    if epoch % 10 == 9:
        print('%d loss: %.5f' %(epoch + 1, running_loss / 10))
        running_loss = 0.0

print('Finished Training')

#save model
torch.save(diagnoser, "ann_model\\bpsk_mbs_detector.pkl")
torch.save(diagnoser.state_dict(), "ann_model\\bpsk_mbs_detector.pkl")

#create two figures
pl.figure(1)
pl.figure(2)

#choose figure 1
pl.figure(1)
pl.plot(np.array(train_loss))
pl.title("Training Loss")
pl.xlabel("Epoch")
pl.ylabel("Loss")

#test
diagnoser.eval()
eval_loss = []
batch2 = 1000
test_len = 1000
for i in range(test_len):
    inputs, labels, _ = mana.random_batch(batch2)
    labels = (torch.sum(labels, 1) > 0).float().view(batch2, 1)
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
