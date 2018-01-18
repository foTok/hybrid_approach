"""
this script is used to generate data
"""
import os
import sys
import numpy as np
from bpsk_generator import Bpsk

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

#Amplifier fault
#BPSK.insert_fault("amplify", 0.1)
#TMA
#BPSK.insert_fault("tma", 0.11)
#TMB
#BPSK.insert_fault("tmb", (8.8 * 10**6, 10 * 10**6))
#pseudo_rate
#BPSK.insert_fault("pseudo_rate", 0.01)
#carrier_rate
#BPSK.insert_fault("carrier_rate", 0.001)
#carrier_leak
#BPSK.insert_fault("carrier_leak", 0.1)

faults = ["amplify", "tma", "tmb", "pseudo_rate", "carrier_rate", "carrier_leak"]
parameters = [0.1, 0.11, (8.8 * 10**6, 10 * 10**6), 0.01, 0.001, 0.1]
for i in range(len(faults)):
    for j in range(i+1, len(faults)):
        time = 0.0001
        fault_time = time / 2
        bpsk = Bpsk()
        bpsk.insert_fault_para(faults[i], parameters[i])
        bpsk.insert_fault_para(faults[j], parameters[j])
        bpsk.insert_fault_time("all", fault_time)
        data = bpsk.generate_signal(time)
        file_name = path + "\\data\\" + str(faults[i]+"_"+faults[j]) + ".npy"
        np.save(file_name, data)
