"""
test if data generate and save function works well
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bpsk_navigate
import numpy as np
import matplotlib.pyplot as pl

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\
           +"/bpsk_navigate/data"

data = np.load(data_path + "/amplify_0.1.npy")
pl.plot(data[:, 3])
pl.show()
