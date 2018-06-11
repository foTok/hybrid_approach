"""
learning structure from data
This file help to learn the influence graph based on the influence coefficients.
Output: Influence Graph
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
from data_manger.utilities import get_file_list
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import heapq
from math import log

#---------------function definition begin--------------#
def set1_is_superset_of_set2(set1, set2):
    """
    set1, set2: numpy.array
    here, superset is TRUE SUPERSET
    If set1 == set2, return False
    """
    if ((set1 - set2) ==  1).any() and\
       ((set1 - set2) != -1).all():
        return True
    return False

def minimal_superset(sets, set2):
    """
    find the minimal super set of set1 in sets
    """
    superset0 = []
    superset1 = []
    if np.sum(set2) == 0:
        return superset1
    for i, set1 in zip(range(len(sets)), sets):
        if set1_is_superset_of_set2(set1, set2):
            superset0.append((set1, i))
    for set10 in superset0:
        break_flag = False
        for set20 in superset0:
            if set1_is_superset_of_set2(set10[0], set20[0]):
                break_flag = True
                break
        if not break_flag:
            superset1.append(set10[1])

    return superset1
#---------------function definition end--------------#


coeff = np.array([[0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [1, 1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1]])

graph = np.zeros((5, 5))

for i in range(5):
    superset = minimal_superset(coeff, coeff[i, :])
    for j in superset:
        graph[i, j] = 1

print("graph=\n", graph)
