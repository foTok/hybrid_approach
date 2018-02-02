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
DATA_PATH = PATH + "\\bpsk_navigate\\data\\"
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
    power_real = sum(output_real**2)/len(output_real)

    noise_power = NOISE_POWER_RATIO*power_real
    noise_std = np.diag(noise_power**0.5)
    noise = ((rand(len(data_real), 4) - 0.5)*2).dot(noise_std)
    output_obs = output_real + noise

    residuals = output_obs - data_pre[:, 1:]

    noise_I_std = np.diag(1/(noise_power**0.5))
    abs_normal_residuals = abs(residuals.dot(noise_I_std))
    Z_test = abs_normal_residuals > 1

    Z_test_all = np.sum(Z_test, axis=1)
    pos = np.where(Z_test_all != 0)[0]
    if len(pos) != 0:
        fault = min(pos - 500, key=abs) + 500
    else:
        fault = 0
    fault_time.append(fault)

pl.figure(1)
pl.plot(fault_time)
pl.show()
