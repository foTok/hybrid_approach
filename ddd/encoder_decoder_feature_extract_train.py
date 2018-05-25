"""
train the encoder decoder feature extractor
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
from ann_diagnoser.encoder_decoder import EncoderDecoder
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from torch.autograd import Variable
from tensorboardX import SummaryWriter


#prepare data
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
ed_name = "ED2.pkl"
basis_name = "ED1.pkl"
step_len=100
criterion = torch.nn.MSELoss()

mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

# ED = EncoderDecoder()
ED = torch.load(ANN_PATH + basis_name)
optimizer = optim.Adam(ED.parameters(), lr=0.001, weight_decay=8e-3)
print(ED)

#train
epoch = 8000
batch = 2000
train_loss = []
running_loss = 0.0
for i in range(epoch):
    inputs, _, _, _ = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
    optimizer.zero_grad()
    outputs = ED(inputs)
    loss = criterion(outputs, inputs)
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
torch.save(ED, ANN_PATH + ed_name)

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
ED_test = torch.load(ANN_PATH + ed_name)
ED_test.eval()
eval_loss = []
batch2 = 1000
epoch2 = 100
for i in range(epoch2):
    inputs, labels, _, _ = mana2.random_batch(batch2, normal=0.2, single_fault=10, two_fault=1)
    outputs = ED_test(inputs)
    loss = criterion(outputs, inputs)
    eval_loss.append(loss.item())

#figure 2
pl.figure(2)
pl.plot(np.array(eval_loss))
pl.title("Random Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("Loss")
pl.show()
