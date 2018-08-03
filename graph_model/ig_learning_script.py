"""
The script to learn IG
"""

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from math import log
from math import exp
from graph_model.ig_learning import IG_learning

#data amount
small_data = True
#settings
snr         = 20
PATH        = parentdir
DATA_PATH   = PATH + "\\bpsk_navigate\\data\\" + ("big_data\\" if not small_data else "small_data\\")
dn          = np.array([2, 2, 10, 10, 10])
fsm         = [[0, 1, 0, 1, 1],
               [0, 1, 0, 1, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1],
               [0, 0, 0, 0, 1],
               [0, 1, 0, 1, 1]]
fsm         = np.array(fsm)

learner = IG_learning()
learner.set_fsm(fsm)
learner.set_discrete_num(dn)
learner.load_normal_data(DATA_PATH, None, snr)
learner.init_queue()
learner.greedy_search()
best, score = learner.best()
print(score)
best.print()
print("DONE")