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

#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)


