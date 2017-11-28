"""
this script is used to generate data
"""

from data_generate import generate_signal

#Amplifier fault
#BPSK.insert_fault("amplify", 0.1)
DELTA_A = 0.1
FILE1 = "data/amplify_" + str(DELTA_A) + ".npy"
generate_signal(FILE1, "amplify", DELTA_A)

#TMA
#BPSK.insert_fault("tma", 0.11)
DELTA_T = 0.11
FILE2 = "data/tma_" + str(DELTA_T) + ".npy"
generate_signal(FILE2, "tma", DELTA_T)

#pseudo_rate
#BPSK.insert_fault("pseudo_rate", 0.01)
DELTA_P = 0.01
FILE3 = "data/pseudo_rate_" + str(DELTA_P) + ".npy"
generate_signal(FILE3, "pseudo_rate", DELTA_P)

#carrier_rate
#BPSK.insert_fault("carrier_rate", 0.001)
DELTA_C = 0.001
FILE4 = "data/carrier_rate_" + str(DELTA_C) + ".npy"
generate_signal(FILE4, "carrier_rate", DELTA_C)

#carrier_leak
#BPSK.insert_fault("carrier_leak", 0.1)
DELTA_L = 0.1
FILE5 = "data/carrier_leak_" + str(DELTA_L) + ".npy"
generate_signal(FILE5, "carrier_leak", DELTA_L)
