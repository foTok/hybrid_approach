"""
the main file to conduct the computation
"""
import os
from ann_diagnoser.diagnoser_full_connect import DiagnoerFullConnect
from data_manger.data_tank import DataTank
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np

from data_manger.utilities import get_file_list


#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
mana = DataTank()

step_len=100
mana.set_fault_type(["tma", "tmb", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify"])
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len)

#set ann fullconnect diagnoser
    #ANN
diagnoser = DiagnoerFullConnect(step_len=mana.step_len())
print(diagnoser)
    #loss function
criterion = nn.MSELoss()
    #optimizer
lr = 0.1
momentum=0.9
optimzer = optim.SGD(diagnoser.parameters(), lr=lr, momentum=momentum)

    #train
episode = 2000
batch = 50

train_loss = []
for epoch in range(episode):
    running_loss = 0.0
    inputs, labels = mana.random_batch(batch)
    #print(inputs)
    inputs, labels = Variable(inputs), Variable(labels)

    optimzer.zero_grad()

    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    #print(outputs)
    #print(labels)
    loss.backward()
    optimzer.step()

    train_loss.append(loss.data[0])

    running_loss += loss.data[0]
    if epoch % 10 == 9:
        print('%d loss: %.5f' %(epoch + 1, running_loss / 10))
        running_loss = 0.0

print('Finished Training')

#create two figures
pl.figure(1)
pl.figure(2)

#choose figure 1
pl.figure(1)
pl.plot(np.array(train_loss))
pl.title("Training Loss")
pl.xlabel("Epoch")
pl.ylabel("MSE Loss")

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
    if loss.data[0] > 0.06:
        print('%d loss: %.5f' %(i + 1, loss.data[0]))
        print(labels)
        print(outputs)

#choose figure 2
pl.figure(2)
pl.plot(np.array(eval_loss))
pl.title("Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("MSE Loss")
pl.show()
