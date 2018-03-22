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
from hybrid_algorithm.a_star import a_star
from mbd.res_Z_test import Z_test

#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
MODEL_PATH = PATH + "\\ann_model\\"
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

pc = [{0, 1, 2}, {3}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}]
model = MODEL_PATH + "bpsk_mbs_isolator.pkl"
hybrid_isolator = a_star(6)
ddm = torch.load(model)


batch = 2000
#diagnosis result
d_r = None
h_r = []
inputs, labels, _, res = mana.random_batch(batch, normal=0,single_fault=10, two_fault=4)
labels = labels.data.numpy()
#Data Driven
priori = ddm(inputs).data.numpy()
d_label = (priori > 0.5)
d_r = [x==y for x,y in zip(d_label, labels)]
d_a = (sum(d_r)/len(d_r))

#Hybrid
#find residuals
Z = Z_test(res, 0.98)
for i in range(len(inputs)):
    the_label = labels[i, :]
    #to debug
    # if the_label[3] == 1:
    #     print("bug may here")
    hybrid_isolator.set_priori(priori[i, :])
    hybrid_isolator.clear_conf_consis()
    Z_i = Z[i,:]
    #to debug
    # if Z_i[1]==False and the_label[3] == 1:
    #     print("bug may here")
    for j in range(len(Z_i)):
        if Z_i[j]==1:
            hybrid_isolator.add_conflict(pc[j])
        else:
            hybrid_isolator.add_consistency(pc[j])
    m_i = hybrid_isolator.most_probable(1)
    best = np.array(m_i[0])
    h_r.append(best[0] == the_label)
h_a = (sum(h_r)/len(h_r))

print("d_a={}".format(d_a))
print("h_a={}".format(h_a))
