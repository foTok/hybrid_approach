"""
this script is used to generate data
"""
import os
import sys
import numpy as np
from bpsk_generator import Bpsk

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))


faults = ["amplify", "tma", "tmb", "pseudo_rate", "carrier_rate", "carrier_leak"]
parameters = [0.09, 0.13, (10.2 * 10**6, 9.8 * 10**6), 0.013, 0.0013, 0.13]
for i in range(len(faults)):
    for j in range(i+1, len(faults)):
        time = 0.0001
        fault_time = time / 2
        bpsk = Bpsk()
        bpsk.insert_fault_para(faults[i], parameters[i])
        bpsk.insert_fault_para(faults[j], parameters[j])
        bpsk.insert_fault_time("all", fault_time)
        data = bpsk.generate_signal(time)
        file_name = path + "\\data\\" + str(faults[i]+"_"+faults[j]) + "3.npy"
        np.save(file_name, data)
