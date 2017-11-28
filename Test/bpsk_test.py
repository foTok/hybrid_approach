"""
test the bpsk module
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as pl
from bpsk_navigate.bpsk import Bpsk

BPSK = Bpsk()

#Amplifier fault
#BPSK.insert_fault("amplify", 1.0)

#TMA
BPSK.insert_fault("tma", 0.3)

#pseudo_rate
#BPSK.insert_fault("pseudo_rate", 1023000*10)

#carrier_rate
#BPSK.insert_fault("carrier_rate", 1023000*10)

#carrier_leak
#BPSK.insert_fault("carrier_leak", 0.1)

#set fault time
BPSK.set_fault_time(0.0001/2)

SIG = BPSK.generate_signal(0.0001)


pl.plot(SIG[:, 3])
pl.show()