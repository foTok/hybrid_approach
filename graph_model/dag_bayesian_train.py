"""
train a DAG Bayesian network
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
import torch
import numpy as np
import matplotlib.pyplot as pl
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.Bayesian_learning import Bayesian_structure
from graph_model.Bayesian_learning import Bayesian_learning
from graph_model.utilities import organise_data

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
FE_PATH = PATH + "\\ann_model\\"
fe_file = "FE0.pkl"
step_len=100
epoch = 2000
batch = 2000

#load fe
FE = torch.load(FE_PATH+fe_file)
FE.eval()

#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

#labels, 6; features, 10 (12-10); residuals 3.
BL = Bayesian_learning(6+10+3)
BL.init_queue()
inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=0)
for i in range(epoch):
    # inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=0)
    feature = FE.fe(inputs)
    batch_data = organise_data(inputs, labels, res, feature)

    # #for test
    # length = len(batch_data)
    # n = len(batch_data[0, :])
    # lb = np.array([np.argwhere(x == 1)[0][0] for x in batch_data[:, :6]])
    # for k in range(13):
    #     sk = batch_data[:, k+6]
    #     pl.figure(k+1)
    #     for i in range(6):
    #         mask = (lb==i)
    #         sk_i = sk[mask]
    #         pl.subplot(6,1,i+1)
    #         pl.hist(sk_i, 30)
    # pl.show()

    # #for test
    # length = len(batch_data)
    # n = len(batch_data[0, :])
    # plt = batch_data[:, 6:]
    # lb = np.array([np.argwhere(x == 1)[0][0] for x in batch_data[:, :6]])
    # pl.figure()
    # for i in range(n-6):
    #     pl.subplot(5, 3, i+1)
    #     pl.scatter(lb, plt[:, i])
    # pl.show()

    BL.set_batch(batch_data)
    BL.step()
    #Add loss to visual
best = BL.best_candidate()
#TODO
#get CPT
#save model