"""
train a DAG Bayesian network
"""
import os
import torch
import numpy as np
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

#labels, 6; features, 12; residuals 3.
BL = Bayesian_learning(6+12+3)
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=1)
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
    res = np.mean(res, axis=2)
    batch_data[:,18:20] = res[:, :2]
    #res3
    inputs = inputs.detach().numpy()
    s3 = np.mean(inputs[:, 3], axis=1)
    s4 = np.mean(inputs[:, 4], axis=1)
    batch_data[:, -1] = ( s4 - 10 * s3)

    BL.set_batch(batch_data)
    BL.step()
    #TODO
    #Add loss to visual
best = BL.best_candidate()
#TODO
#get CPT
#save model