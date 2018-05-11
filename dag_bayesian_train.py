"""
train a DAG Bayesian network
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as pl
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.Bayesian_learning import Bayesian_structure
from graph_model.Bayesian_learning import Bayesian_learning

#settings
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
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
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=0)
    feature = FE.fe(inputs)
    length = len(feature)
    batch_data = np.zeros((length, 6+12+3))
    #the first 6 colums(0:6) are fault labels
    #["tma", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify", "tmb"]
    fault_labels = labels.detach().numpy()
    batch_data[:,:6] = fault_labels
    #the mid 12 colums(6:18) are features
    feature = feature.detach().numpy()
    batch_data[:, 6:18] = feature
    #the last 3 colums(18:21) are residuals
    res = np.array(res)
    #res12
    res = np.mean(np.abs(res), axis=2)
    batch_data[:,18:20] = res[:, :2]
    #res3
    inputs = inputs.detach().numpy()
    # s3 = np.mean(inputs[:, 3], axis=1)
    # s4 = np.mean(inputs[:, 4], axis=1)
    # batch_data[:, -1] = ( s4 - 10 * s3)
    s3 = inputs[:, 3]
    s4 = inputs[:, 4]
    batch_data[:, -1] = np.mean(np.abs( s4 - 10 * s3), axis=1)
    # #for test
    # plt = np.zeros((length, 4))
    # plt[:, 0:3] = batch_data[:,-3:]
    # lb = np.array([np.argwhere(x == 1)[0][0] for x in batch_data[:, :6]])
    # pl.figure(1)
    # pl.scatter(lb, plt[:, 0])
    # pl.figure(2)
    # pl.scatter(lb, plt[:, 1])
    # pl.figure(3)
    # pl.scatter(lb, plt[:, 2])
    # pl.show()
    
    #data analysis
    # var = np.var(batch_data, axis=0)
    # print("var=", var)

    a1 = batch_data[:, :7]
    a2 = batch_data[:, 8:17]
    a3 = batch_data[:, 18:]
    real_data = np.concatenate((a1,a2,a3), axis=1)

    BL.set_batch(real_data)
    BL.step()
    #Add loss to visual
best = BL.best_candidate()
#TODO
#get CPT
#save model