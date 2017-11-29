"""
the main file to conduct the computation
"""

from ann_diagnoser.diagnoser_full_connect import DiagnoerFullConnect
from data_manger.data_tank import DataTank
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#prepare data
data_path = "bpsk_navigate/data"
mana = DataTank()

step_len=100
mana.set_fault_type(["amplify", "tma", "pseudo_rate", "carrier_rate", "carrier_leak"])
mana.read_data(data_path+"/amplify_0.1.npy", fault_type="amplify", step_len=step_len)
mana.read_data(data_path+"/carrier_leak_0.1.npy", fault_type="carrier_leak", step_len=step_len)
mana.read_data(data_path+"/carrier_rate_0.001.npy", fault_type="carrier_rate", step_len=step_len)
mana.read_data(data_path+"/pseudo_rate_0.01.npy", fault_type="pseudo_rate", step_len=step_len)
mana.read_data(data_path+"/tma_0.11.npy", fault_type="tma", step_len=step_len)


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

    running_loss += loss.data[0]
    if epoch % 10 == 9:
        print('%d loss: %.5f' %(epoch + 1, running_loss / 10))
        running_loss = 0.0

print('Finished Training')

#test
diagnoser.eval()
test_len = 100
for i in range(test_len):
    inputs, labels = mana.random_batch(1)
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    print('%d loss: %.5f' %(i + 1, loss.data[0]))
    print(outputs)