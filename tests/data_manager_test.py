"""
test if the data manager works well
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_manger.data_tank import DataTank
import numpy as np
import matplotlib.pyplot as pl

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\
           +"/bpsk_navigate/data"

mana = DataTank()

mana.set_fault_type(["amplify", "tma", "pseudo_rate", "carrier_rate", "carrier_leak"])
mana.read_data(data_path+"/amplify_0.1.npy", fault_type="amplify")
mana.read_data(data_path+"/carrier_leak_0.1.npy", fault_type="carrier_leak")
mana.read_data(data_path+"/carrier_rate_0.001.npy", fault_type="carrier_rate")
mana.read_data(data_path+"/pseudo_rate_0.01.npy", fault_type="pseudo_rate")
mana.read_data(data_path+"/tma_0.11.npy", fault_type="tma")

chosen_data0 = mana.choose_data_randomly([0, 0, 0, 0, 0], 10)
chosen_data1 = mana.choose_data_randomly([1, 0, 0, 0, 0], 10)
chosen_data2 = mana.choose_data_randomly([0, 1, 0, 0, 0], 10)
chosen_data3 = mana.choose_data_randomly([0, 0, 1, 0, 0], 10)
chosen_data4 = mana.choose_data_randomly([0, 0, 0, 1, 0], 10)
chosen_data5 = mana.choose_data_randomly([0, 0, 0, 0, 1], 10)
chosen_data6 = mana.choose_data_randomly([1, 1, 0, 0, 0], 10)


input_data, target = mana.random_batch(10)

print("test end!")