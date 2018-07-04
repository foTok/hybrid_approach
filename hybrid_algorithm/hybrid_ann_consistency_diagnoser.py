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
    def __init__(self, alpha=0.95):
        super(hybrid_ann_consistency_diagnoser, self).__init__()
        self.res_set         = []
        self.res_values      = []
        self.alpha           = alpha

    def set_res_set(self, res_set):
        """
        RT
        """
        self.res_set = res_set

    def set_res_values(self,  res_values):
        """
        RT
        """
        self.res_values = res_values

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

    def __conflict_cost(self, candidate):
        """
        cost based on conflict
        """
        cost = 0
        for i in range(len(self.res_set)):
            if not self.res_values[i]:
                conf   = self.res_set[i]
                cost_c = -log(self.alpha) if self.__solve_conflict(candidate, conf) else -log(1 - self.alpha)
                cost   = cost + cost_c
        return cost

    def __hold_consistency(self, candidate, consis):
        """
        check if candidate holds consis
        """
        candi = np.array([-1]*len(self.order))
        for i in range(len(candidate)):
            id          = self.order[i]
            candi[id]   = candidate[i]
        for i in range(len(consis)):
            if (consis[i] == 1) and (candi[i] == 1):
                #consis[i]==1 means fault i is included in the consistency
                #So, candi[i] must be 0 or -1. 1 means the consistency is borken
                return False
        return True

    def __consistency_cost(self, candidate):
        """
        cost based on consistency
        """
        alpha = 0.1
        beta = (1 - 2*alpha) * self.alpha + alpha
        cost = 0
        for i in range(len(self.res_set)):
            if self.res_values[i]:
                consis = self.res_set[i]
                cost_c = -log(beta) if self.__hold_consistency(candidate, consis) else -log(1 - beta)
                cost   = cost + cost_c
        return cost

    def __likelihood_cost(self, candidate):
        """
        likelihood cost
        """
        cost_conf   = self.__conflict_cost(candidate)
        cost_consis = self.__consistency_cost(candidate)
        cost        = cost_conf + cost_consis
        return cost

    def _cost(self, candidate):
        """
        compute the cost of candidate
        """
        alpha = 1e-3
        cost = 0
        #priori cost
        for c, i in zip(candidate, range(len(candidate))):
            id = self.order[i]
            value = c
            p_c = self.priori[id][value]
            p_c = (1 - 2*alpha) * p_c + alpha    #shrink
            cost_c = -log(p_c)
            cost = cost + cost_c
        #add likelihood cost
        cost = cost + self.__likelihood_cost(candidate)
        return cost
