"""
This file is used to conduct hybrid seach based on features and residuals
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import numpy as np
from scipy import stats
from hybrid_algorithm.a_star_frame import a_star_frame
from hybrid_algorithm.utilities import hypothesis_test as obs_pro
from math import log

class hybrid_search(a_star_frame):
    """
    hybrid diagnosis
    operator, |entity|
    |data|--->|data driven isolator|--->belief2pro--->|initial probability|
       |                                                      |
       |                                                      |
       |                                                      |
    |feature extractor      |           |graph model|         |
    |and residual generator |                 |               |
       |                                      |               |
       |                                      |               |
       |                                      |               |
    |features and residuals|--------------------------->|A* searach|
    The key of implement A* search is to implement the cost function based on
    |graph model|, |features and residuals| and |initial probability|
    """
    def __init__(self):
        super(hybrid_search, self).__init__()
        self.graph_model = None

    def set_graph_model(self, model):
        """
        set the graph_model
        """
        self.graph_model = model

    def add_obs(self, id, obs):
        """
        add obs to model
        """
        self.graph_model.parameters.add_obs_ass(id, obs)

    def cost(self, candidate):
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
        #fml cost
        self.graph_model.struct.set_skip()
        for fml in self.graph_model.struct:
            cost_fml = self.fml_cost(fml, candidate)
            cost = cost + cost_fml
        return cost            

    def fml_cost(self, fml, candidate):
        """
        compute the cost of fml based on the candidate and obs
        if not all the values in the fml is assigned, return 0
        """
        #To avoid numeric problems
        alpha = 1e-20
        #check if all the variables in fml are assgined
        for i in fml:
            if (not self.candidate_has(candidate, i)) and\
               (i not in self.graph_model.parameters.obs_ass):
                return 0
        #now we should compute the cost
        parents = fml[:-1]
        kid = fml[-1]
        X = np.zeros((1, len(parents)))
        for x, i in zip(parents, range(len(parents))):
            if x < 6: #0...6 are faults
                index_x = self.order.index(x)
                value_x = candidate[index_x]
                X[0, i] = value_x
            else:     #features or residuals
                value_x = self.graph_model.parameters.obs_ass[x]
                X[0, i] = value_x
        #Y must be a feature or residual
        value_y = self.graph_model.parameters.obs_ass[kid]
        Y = np.array([value_y])
        beta, var = self.graph_model.parameters.fml_tank[fml]
        p = obs_pro(X, Y, beta, var)
        #To avoid numeric problems
        p = (p + alpha) / (1+alpha)
        cost = -log(p)
        return cost

    def candidate_has(self, candidate, i):
        """
        check if the candidate has varialbe i
        """
        for c_i in range(len(candidate)):
            if i == self.order[c_i]:
                return True
        return False

