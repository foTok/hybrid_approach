"""
this script is used to generate data
"""
import os
import sys
from data_generator import generate_signal

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

#Amplifier fault
#BPSK.insert_fault("amplify", 0.1)
DELTA_A = 0.1
FILE1 = "\\data\\amplify_" + str(DELTA_A) + ".npy"
generate_signal(path + FILE1, "amplify", DELTA_A)

#TMA
#BPSK.insert_fault("tma", 0.11)
DELTA_T = 0.11
FILE2 = "\\data\\tma_" + str(DELTA_T) + ".npy"
generate_signal(path + FILE2, "tma", DELTA_T)

#TMB
#BPSK.insert_fault("tmb", (8.8 * 10**6, 10 * 10**6))
DELTA_F = (8.8 * 10**6, 10 * 10**6)
FILE3 = "\\data\\tmb_" + str(DELTA_F) + ".npy"
generate_signal(path + FILE3, "tmb", DELTA_F)

#pseudo_rate
#BPSK.insert_fault("pseudo_rate", 0.01)
DELTA_P = 0.01
FILE4 = "\\data\\pseudo_rate_" + str(DELTA_P) + ".npy"
generate_signal(path + FILE4, "pseudo_rate", DELTA_P)

#carrier_rate
#BPSK.insert_fault("carrier_rate", 0.001)
DELTA_C = 0.001
FILE5 = "\\data\\carrier_rate_" + str(DELTA_C) + ".npy"
generate_signal(path + FILE5, "carrier_rate", DELTA_C)

#carrier_leak
#BPSK.insert_fault("carrier_leak", 0.1)
DELTA_L = 0.1
FILE6 = "\\data\\carrier_leak_" + str(DELTA_L) + ".npy"
generate_signal(path + FILE6, "carrier_leak", DELTA_L)
