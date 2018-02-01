"""
this file is used to test the performance of fcann diagnoser
"""

import os
from ann_diagnoser.diagnoser_full_connect import DiagnoerFullConnect
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np

#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
MODEL_PATH = PATH + "\\ann_model\\"
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len)

#set ann fullconnect diagnoser
    #ANN
diagnoser = DiagnoerFullConnect(step_len=mana.step_len())
print(diagnoser)
diagnoser.load_state_dict(torch.load(MODEL_PATH+"bpsk_fc_params.pkl"))

criterion = nn.MSELoss()

#test
diagnoser.eval()
eval_loss = []
test_len = 1000
for i in range(test_len):
    inputs, labels = mana.random_batch(1)
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    eval_loss.append(loss.data[0])
    if loss.data[0] > 0.04:
        print('%d loss: %.5f' %(i + 1, loss.data[0]))
        print(labels)
        print(outputs)

#create a figure
pl.figure(1)
pl.plot(np.array(eval_loss))
pl.title("Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("MSE Loss")
pl.show()
