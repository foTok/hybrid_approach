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

class hybrid_ann_consistency_diagnoser(a_star_frame):
    """
    the hybrid diagnoser just based on hybrid ann diagnoser
    """
    def __init__(self, p0=0.999):
        super(hybrid_ann_consistency_diagnoser, self).__init__()
        self.conflict_set   = []
        self.p0             = p0

    def add_conflicts(self, conflicts):
        """
        add conflicts into self.conflict_set
        """
        self.conflict_set.clear()
        for conf in conflicts:
            self.conflict_set.append(conf)

    def __solve_conflict(self, candidate, conf):
        """
        check if candidate solves conf
        """
        candi = np.array([-1]*len(self.order))
        for i in range(len(candidate)):
            id          = self.order[i]
            candi[id]   = candidate[i]
        for i in range(len(conf)):
            if (conf[i] == 1) and (candi[i] != 0):
                #If there is no conflict variable that no set by candidate or the set as 1.
                #The conf is assumped to be or will to be solved
                return True
        return False

    def __solve_all_conflicts(self, candidate):
        """
        Check if candidate solves all the conflicts.
        """
        for conf in self.conflict_set:
            if not self.__solve_conflict(candidate, conf):
                return False
        return True

    def _cost(self, candidate):
        """
        compute the cost of candidate
        """
        alpha = 1e-20
        #initial cost
        cost = -log(self.p0) if self.__solve_all_conflicts(candidate) else -log(1 - self.p0)
        #priori cost
        for c, i in zip(candidate, range(len(candidate))):
            id = self.order[i]
            value = c
            p_c = self.priori[id][value]
            p_c = (p_c + alpha) /(1 + alpha)
            cost_c = -log(p_c)
            cost = cost + cost_c
        return cost
