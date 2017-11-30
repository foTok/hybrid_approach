"""
test the bpsk module
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as pl
from bpsk_navigate.bpsk_generator import Bpsk

BPSK = Bpsk()

#Amplifier fault
#BPSK.insert_fault_para("amplify", 0.1)

#TMA
#BPSK.insert_fault_para("tma", 0.11)

#TMB
#BPSK.insert_fault_para("tmb", (8.8 * 10**6, 10 * 10**6))

#pseudo_rate
#BPSK.insert_fault_para("pseudo_rate", 0.01)

#carrier_rate
#BPSK.insert_fault_para("carrier_rate", 0.001)

#carrier_leak
#BPSK.insert_fault_para("carrier_leak", 0.1)

#set fault time
BPSK.insert_fault_time("all", 0.0001/2)

SIG = BPSK.generate_signal(0.0001)


pl.plot(SIG[:, 3])
pl.show()
