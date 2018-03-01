"""
This file diagnosis BPSK system by MBD
"""

import os
import numpy as np
import matplotlib.pyplot as pl
from scipy import stats
from numpy.random import rand
from bpsk_navigate.bpsk_generator import Bpsk
from data_manger.bpsk_data_tank import parse_filename
from data_manger.utilities import get_file_list


#prepare data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
SNR = 20
NOISE_POWER_RATIO = 1/(10**(SNR/10))
TIME = 0.0001

fault_time = []
list_files = get_file_list(DATA_PATH)
for the_file in list_files:
    print(the_file)
    data_real = np.load(DATA_PATH+the_file)
    bpsk = Bpsk()
    data_pre = bpsk.generate_signal_with_input(0.0001, data_real[:, 0])
    #energy
    output_real = data_real[:, 1:]
    signal_power = np.var(output_real, 0)

    noise_power = NOISE_POWER_RATIO*signal_power
    noise_weight = noise_power**0.5
    noise = np.random.normal(0, 1, [len(output_real), len(output_real[0])]) * noise_weight
    output_obs = output_real + noise
    residuals = output_obs - data_pre[:, 1:]

    noise_I_std = np.diag(1/(noise_power**0.5))
    abs_normal_residuals = abs(residuals.dot(noise_I_std))
    Z_test = abs_normal_residuals > 3

    Z_test_all = np.sum(Z_test, axis=1)
    pos = np.where(Z_test_all != 0)[0]
    pos = [i for i in pos if i > 500 ]
    fault = min(pos)
    fault_time.append(fault)

pl.figure(1)
pl.plot(fault_time)
pl.show()
