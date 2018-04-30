"""
train a DAG Bayesian network
"""
import os
import torch
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from graph_model.Bayesian_learning import Bayesian_structure
from graph_model.Bayesian_learning import Bayesian_learning

#settings
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
FE_PATH = PATH + "\\ann_model\\"
fe_file = "bpsk_mbs_isolator1.pkl"
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
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

#there are 20 features and 3 residuals
BL = Bayesian_learning(20+3)
for i in range(epoch):
    inputs, labels, _, res = mana.random_batch(batch, normal=0, single_fault=10, two_fault=1)
    feature = FE.fe(inputs)
    #TODO
    #convert features to real features
    #get real residuals
    #set batch data
    batch_data = None
    BL.set_batch(batch_data)
    BL.step()
    #TODO
    #Add loss to visual
best = BL.best_candidate()
#TODO
#get CPT
#save model