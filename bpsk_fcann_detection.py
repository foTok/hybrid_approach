"""
use full connection ANN to detect fault
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

def fault_probability(fault_vec):
    p = 1.0
    for i in fault_vec:
        p = p*(1-i)
    return 1 - p

step_len=100
#set ann fullconnect diagnoser
    #ANN
diagnoser = DiagnoerFullConnect(step_len=step_len*5)
print(diagnoser)
diagnoser.load_state_dict(torch.load(MODEL_PATH+"bpsk_fc_params_mse_reg_2000.pkl"))
diagnoser.eval()

fault_time = []
list_files = get_file_list(DATA_PATH)
for the_file in list_files:
    print(the_file)
    count = 0
    data = np.load(DATA_PATH + the_file)
    for i in range(len(data)-step_len):
        x = data[i:i+step_len, :]
        x = [i for sublist in x for i in sublist]
        x = Variable(torch.Tensor(x))
        y = diagnoser(x)
        #print(y)
        p = fault_probability(y.data)
        if p>0.55:
            count = count + 1
            if i+step_len>500:
                print("false alarm = %d" %count)
                fault_time.append(i+step_len)
                break

#create a figure
pl.figure(1)
pl.plot(fault_time)
pl.show()
