"""
the main file to conduct the computation
"""
import os
import sys
from ann_diagnoser.diagnoser_full_connect import DiagnoerFullConnect
from data_manger.data_tank import DataTank
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np


#prepare data
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
data_path = path + "/bpsk_navigate/data"
mana = DataTank()
mana2 = DataTank()

step_len=100
mana.set_fault_type(["amplify", "tma", "tmb", "pseudo_rate", "carrier_rate", "carrier_leak"])
#mana.read_data(data_path+"/amplify_0.1.npy", fault_type="amplify", step_len=step_len)
#mana.read_data(data_path+"/carrier_leak_0.1.npy", fault_type="carrier_leak", step_len=step_len)
#mana.read_data(data_path+"/carrier_rate_0.001.npy", fault_type="carrier_rate", step_len=step_len)
#mana.read_data(data_path+"/pseudo_rate_0.01.npy", fault_type="pseudo_rate", step_len=step_len)
#mana.read_data(data_path+"/tma_0.11.npy", fault_type="tma", step_len=step_len)
#mana.read_data(data_path+"/tmb_(8800000.0, 10000000).npy", fault_type="tmb", step_len=step_len)
mana.read_data(data_path+"/amplify_tma.npy", fault_type=["amplify", "tma"], step_len=step_len)
mana.read_data(data_path+"/amplify_tmb.npy", fault_type=["amplify", "tmb"], step_len=step_len)
mana.read_data(data_path+"/amplify_carrier_leak.npy", fault_type=["amplify", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/amplify_carrier_rate.npy", fault_type=["amplify", "carrier_rate"], step_len=step_len)
mana.read_data(data_path+"/amplify_pseudo_rate.npy", fault_type=["amplify", "pseudo_rate"], step_len=step_len)
mana.read_data(data_path+"/carrier_rate_carrier_leak.npy", fault_type=["carrier_rate", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/pseudo_rate_carrier_leak.npy", fault_type=["pseudo_rate", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/pseudo_rate_carrier_rate.npy", fault_type=["pseudo_rate", "carrier_rate"], step_len=step_len)
mana.read_data(data_path+"/tma_carrier_leak.npy", fault_type=["tma", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/tma_carrier_rate.npy", fault_type=["tma", "carrier_rate"], step_len=step_len)
mana.read_data(data_path+"/tma_pseudo_rate.npy", fault_type=["tma", "pseudo_rate"], step_len=step_len)
mana.read_data(data_path+"/tma_tmb.npy", fault_type=["tma", "tmb"], step_len=step_len)
mana.read_data(data_path+"/tmb_carrier_leak.npy", fault_type=["amplify", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/tmb_carrier_rate.npy", fault_type=["amplify", "carrier_rate"], step_len=step_len)
mana.read_data(data_path+"/tmb_pseudo_rate.npy", fault_type=["amplify", "pseudo_rate"], step_len=step_len)
mana.read_data(data_path+"/amplify_tma2.npy", fault_type=["amplify", "tma"], step_len=step_len)
mana.read_data(data_path+"/amplify_tmb2.npy", fault_type=["amplify", "tmb"], step_len=step_len)
mana.read_data(data_path+"/amplify_carrier_leak2.npy", fault_type=["amplify", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/amplify_carrier_rate2.npy", fault_type=["amplify", "carrier_rate"], step_len=step_len)
mana.read_data(data_path+"/amplify_pseudo_rate2.npy", fault_type=["amplify", "pseudo_rate"], step_len=step_len)
mana.read_data(data_path+"/carrier_rate_carrier_leak2.npy", fault_type=["carrier_rate", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/pseudo_rate_carrier_leak2.npy", fault_type=["pseudo_rate", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/pseudo_rate_carrier_rate2.npy", fault_type=["pseudo_rate", "carrier_rate"], step_len=step_len)
mana.read_data(data_path+"/tma_carrier_leak2.npy", fault_type=["tma", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/tma_carrier_rate2.npy", fault_type=["tma", "carrier_rate"], step_len=step_len)
mana.read_data(data_path+"/tma_pseudo_rate2.npy", fault_type=["tma", "pseudo_rate"], step_len=step_len)
mana.read_data(data_path+"/tma_tmb2.npy", fault_type=["tma", "tmb"], step_len=step_len)
mana.read_data(data_path+"/tmb_carrier_leak2.npy", fault_type=["tmb", "carrier_leak"], step_len=step_len)
mana.read_data(data_path+"/tmb_carrier_rate2.npy", fault_type=["tmb", "carrier_rate"], step_len=step_len)
mana.read_data(data_path+"/tmb_pseudo_rate2.npy", fault_type=["tmb", "pseudo_rate"], step_len=step_len)

mana2.set_fault_type(["amplify", "tma", "tmb", "pseudo_rate", "carrier_rate", "carrier_leak"])
mana2.read_data(data_path+"/amplify_0.1.npy", fault_type="amplify", step_len=step_len)
mana2.read_data(data_path+"/carrier_leak_0.1.npy", fault_type="carrier_leak", step_len=step_len)
mana2.read_data(data_path+"/carrier_rate_0.001.npy", fault_type="carrier_rate", step_len=step_len)
mana2.read_data(data_path+"/pseudo_rate_0.01.npy", fault_type="pseudo_rate", step_len=step_len)
mana2.read_data(data_path+"/tma_0.11.npy", fault_type="tma", step_len=step_len)
mana2.read_data(data_path+"/tmb_(8800000.0, 10000000).npy", fault_type="tmb", step_len=step_len)
#mana2.read_data(data_path+"/amplify_tma3.npy", fault_type=["amplify", "tma"], step_len=step_len)
#mana2.read_data(data_path+"/amplify_tmb3.npy", fault_type=["amplify", "tmb"], step_len=step_len)
#mana2.read_data(data_path+"/amplify_carrier_leak3.npy", fault_type=["amplify", "carrier_leak"], step_len=step_len)
#mana2.read_data(data_path+"/amplify_carrier_rate3.npy", fault_type=["amplify", "carrier_rate"], step_len=step_len)
#mana2.read_data(data_path+"/amplify_pseudo_rate3.npy", fault_type=["amplify", "pseudo_rate"], step_len=step_len)
#mana2.read_data(data_path+"/carrier_rate_carrier_leak3.npy", fault_type=["carrier_rate", "carrier_leak"], step_len=step_len)
#mana2.read_data(data_path+"/pseudo_rate_carrier_leak3.npy", fault_type=["pseudo_rate", "carrier_leak"], step_len=step_len)
#mana2.read_data(data_path+"/pseudo_rate_carrier_rate3.npy", fault_type=["pseudo_rate", "carrier_rate"], step_len=step_len)
#mana2.read_data(data_path+"/tma_carrier_leak3.npy", fault_type=["tma", "carrier_leak"], step_len=step_len)
#mana2.read_data(data_path+"/tma_carrier_rate3.npy", fault_type=["tma", "carrier_rate"], step_len=step_len)
#mana2.read_data(data_path+"/tma_pseudo_rate3.npy", fault_type=["tma", "pseudo_rate"], step_len=step_len)
#mana2.read_data(data_path+"/tma_tmb3.npy", fault_type=["tma", "tmb"], step_len=step_len)
#mana2.read_data(data_path+"/tmb_carrier_leak3.npy", fault_type=["tmb", "carrier_leak"], step_len=step_len)
#mana2.read_data(data_path+"/tmb_carrier_rate3.npy", fault_type=["tmb", "carrier_rate"], step_len=step_len)
#mana2.read_data(data_path+"/tmb_pseudo_rate3.npy", fault_type=["tmb", "pseudo_rate"], step_len=step_len)

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
test_len = 100
for i in range(test_len):
    inputs, labels = mana2.random_batch(1)
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    eval_loss.append(loss.data[0])
    if loss.data[0] > 0.15:
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
