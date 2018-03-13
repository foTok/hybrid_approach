"""
train the ANN predict the residuals
"""
import os
from ann_diagnoser.bpsk_mbd_fc_residual_predictor import res_predictor
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ann_diagnoser.loss_function import PRO
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
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)


r1 = res_predictor(3)
r2 = res_predictor(1)
r3 = res_predictor(5)
r4 = res_predictor(6)

print(r1)
print(r2)
print(r3)
print(r4)

criterion = MSE
optimizer1 = optim.Adam(r1.parameters(), lr=0.005, weight_decay=1.8e-3)
optimizer2 = optim.Adam(r2.parameters(), lr=0.005, weight_decay=1.8e-3)
optimizer3 = optim.Adam(r3.parameters(), lr=0.005, weight_decay=1.8e-3)
optimizer4 = optim.Adam(r4.parameters(), lr=0.005, weight_decay=1.8e-3)

#train
epoch = 2000
batch = 2000
train_loss = np.zeros([epoch, 4])
running_loss = np.zeros(4)
for i in range(epoch):
    _, inputs, _, labels = mana.random_batch(batch, normal=0, single_fault=10, two_fault=4)
    labels = Variable(torch.Tensor(labels))
    labels = torch.mean(labels, dim = 2)
    in1 = inputs[:, 0:3]
    in2 = inputs[:, 3:4]
    in3 = inputs[:, 0:5]
    in4 = inputs[:, :]

    output1 = r1(in1)
    output2 = r2(in2)
    output3 = r3(in3)
    output4 = r4(in4)

    label1 = labels[:, 0:1]
    label2 = labels[:, 1:2]
    label3 = labels[:, 2:3]
    label4 = labels[:, 3:4]

    loss1 = criterion(output1, label1)
    loss2 = criterion(output2, label2)
    loss3 = criterion(output3, label3)
    loss4 = criterion(output4, label4)

    loss1.backward()
    loss2.backward()
    loss3.backward()
    loss4.backward()

    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()

    loss = np.array([loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0]])
    running_loss += loss
    train_loss[i] = loss
    if i % 10 == 9:
        print("{} loss: {}".format(i + 1, running_loss / 10))
        running_loss = np.zeros(4)
print('Finished Training')

#save model
torch.save(r1, "ann_model\\r1.pkl")
torch.save(r1.state_dict(), "ann_model\\r1_para.pkl")
torch.save(r2, "ann_model\\r2.pkl")
torch.save(r2.state_dict(), "ann_model\\r2_para.pkl")
torch.save(r3, "ann_model\\r3.pkl")
torch.save(r3.state_dict(), "ann_model\\r3_para.pkl")
torch.save(r4, "ann_model\\r4.pkl")
torch.save(r4.state_dict(), "ann_model\\r4_para.pkl")

#figure 1
pl.figure(1)
pl.plot(train_loss)
pl.title("Training Loss")
pl.xlabel("Epoch")
pl.ylabel("Loss")

#test
TEST_DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
mana2 = BpskDataTank()
list_files2 = get_file_list(TEST_DATA_PATH)
for file in list_files2:
    mana2.read_data(TEST_DATA_PATH+file, step_len=step_len, snr=20)
r1 = torch.load("ann_model\\r1.pkl")
r2 = torch.load("ann_model\\r2.pkl")
r3 = torch.load("ann_model\\r3.pkl")
r4 = torch.load("ann_model\\r4.pkl")
r1.eval()
r2.eval()
r3.eval()
r4.eval()
batch2 = 1000
epoch2 = 1000
eval_loss = np.zeros([epoch, 4])
for i in range(epoch2):
    _, inputs, _, labels = mana.random_batch(batch, normal=0, single_fault=10, two_fault=4)
    labels = Variable(torch.Tensor(labels))
    labels = torch.mean(labels, dim = 2)
    in1 = inputs[:, 0:3]
    in2 = inputs[:, 3:4]
    in3 = inputs[:, 0:5]
    in4 = inputs[:, :]

    output1 = r1(in1)
    output2 = r2(in2)
    output3 = r3(in3)
    output4 = r4(in4)

    label1 = labels[:, 0:1]
    label2 = labels[:, 1:2]
    label3 = labels[:, 2:3]
    label4 = labels[:, 3:4]

    loss1 = criterion(output1, label1)
    loss2 = criterion(output2, label2)
    loss3 = criterion(output3, label3)
    loss4 = criterion(output4, label4)

    eval_loss[i] = np.array([loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0]])
#figure 2
pl.figure(2)
pl.plot(eval_loss)
pl.title("Random Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("Loss")
pl.show()
