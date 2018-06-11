"""
This file is used to conduct hybrid seach based on hybrid ann
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import numpy as np
from scipy import stats
from hybrid_algorithm.a_star_frame import a_star_frame
from math import log

class hybrid_ann_diagnoser(a_star_frame):
    """
    the hybrid diagnoser just based on hybrid ann diagnoser
    """
    def __init__(self):
        super(hybrid_ann_diagnoser, self).__init__()

    def _cost(self, candidate):
        """
        compute the cost of candidate
        """
        #initial cost
        cost = 0
        #priori cost
        for c, i in zip(candidate, range(len(candidate))):
            id = self.order[i]
            value = c
            p_c = self.priori[id][value]
            cost_c = -log(p_c)
            cost = cost + cost_c
        return cost
