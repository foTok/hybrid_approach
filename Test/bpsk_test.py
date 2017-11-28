"""
test the bpsk module
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as pl
from bpsk_navigate.bpsk import Bpsk
print("imported bpsk")
BPSK = Bpsk()
print("instant bpsk")
#no fault
SIG1 = BPSK.generate_signal(0.0001)

pl.plot(SIG1[:, 2])
pl.show()
