"""
dynamic training means sampleing data based on different paramemeters dynamicly and training diagnoser
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pl
import numpy as np
from bpsk_navigate.pseudo_generator import Pseudo
from data_manger.bpsk_data_tank import BpskDataTank
from mbd.utilities import sample_parameters

#settings
N               = 10
fault_type      = ["tma", "pseudo_rate", "tmb"]
rang            = [[0.2, (0.8 * 10**6, 7.3 * 10**6), -0.05], [0.9, (8.8 * 10**6, 13 * 10**6), 0.05]]
pref            = 3 #1->single-fault, 2->two-fault, 3->single-,two-fault
loss            = 0.4
diagnoser       = None  #TODO
training_data   = BpskDataTank()
para_set        = {}
DATA_PATH       = parentdir + "\\mbd\\data\\"

while True:
    file_list  = []
    parameters = sample_parameters(N, fault_type, rang, pref, para_set)
    for para in parameters:
        simulator = Pseudo()
        #TODO