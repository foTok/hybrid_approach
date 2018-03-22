"""
This file finding probability of residuals for fault isolation
"""


import os
import numpy as np
import torch
import matplotlib.pyplot as pl
from numpy.random import rand
from bpsk_navigate.bpsk_generator import Bpsk
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.bpsk_data_tank import parse_filename
from data_manger.utilities import get_file_list
from data_manger.utilities import statistic
from mbd.res_Z_test import Z_test

#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)
mana.info()


#train
epoch = 2000
batch = 2000

cpt = []
for i in range(epoch):
    