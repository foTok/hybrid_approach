"""
the main file to conduct the computation
"""
from ann_diagnoser.oau_diagnoser_full_connect import DiagnoerFullConnect
from data_manger.data_oau import DataOAU
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
from scipy import stats


alpha = 0.99
x = stats.norm
thresh = x.ppf(1-(1-alpha)/2)

#settings
random_test = 0
unsampled_test = 1
#prepare data
file_name = 'OAU/OAU1201.csv'
mana = DataOAU()
mana.read_data(file_name)
mana.set_test(0.1)


#set ann fullconnect diagnoser
    #ANN
in_size, out_size = mana.data_size()
diagnoser = DiagnoerFullConnect(in_size, out_size)
print(diagnoser)
    #loss function
criterion = nn.MSELoss()
    #optimizer
lr = 0.1
momentum=0.9
optimzer = optim.SGD(diagnoser.parameters(), lr=lr, momentum=momentum)

    #train
episode = 10000
batch = 10

train_loss = []
for epoch in range(episode):
    running_loss = 0.0
    inputs, labels, _ = mana.random_batch(batch)
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
if random_test == 1:
    test_len = 1000
if unsampled_test == 1:
    test_inputs, test_outputs, n_residuals = mana.unsampled_data()
    test_len = len(test_inputs)

for i in range(test_len):
    if random_test == 1:
        inputs, labels, _ = mana.random_batch(1)
    if unsampled_test == 1:
        inputs, labels = test_inputs[i], test_outputs[i]
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = diagnoser(inputs)
    loss = criterion(outputs, labels)
    eval_loss.append(loss.data[0])
    if loss.data[0] > 0.1:
        r = n_residuals[i]
        z0 = [abs(i) > thresh for i in r]
        z = 0 if sum(z0) == 0 else 1
        print('%d loss: %.5f' %(i + 1, loss.data[0]))
        print(outputs)
        print(labels)
        print("r={}".format(str(z)))

#choose figure 2
pl.figure(2)
pl.plot(np.array(eval_loss))
pl.title("Evaluation Loss")
pl.xlabel("Sample")
pl.ylabel("MSE Loss")
pl.show()
