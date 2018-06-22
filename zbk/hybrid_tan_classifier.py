"""
TAN classifier for one model
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from hybrid_algorithm.utilities import hypothesis_test as obs_pro
from math import log
from math import exp

class tan_classifier:
    """
    RT
    """
    def __init__(self):
        self.priori     = None
        self.tan        = None

    def set_priori(self, priori):
        """
        RT
        """
        self.priori     = priori

    def set_tan(self, tan):
        """
        RT
        """
        self.tan        = tan

    def add_obs(self, id_obs_list):
        """
        add obs into it
        """
        self.tan.parameters.clear_obs_ass()
        for id, obs in id_obs_list:
            self.tan.parameters.add_obs_ass(id, obs)

    def normal_pro(self):
        """
        compute the normal probability
        """
        l0 = self.joint_likelihood(0)
        l1 = self.joint_likelihood(1)
        p  = exp(l0) / (exp(l0) + exp(l1))
        return p

    def joint_likelihood(self, f):
        """
        The joint probability when there is no fault
        """
        #To avoid numeric problems
        alpha = 1e-20
        likeli = log(self.priori[f] + alpha)
        for fml in self.tan.struct:
            if len(fml) <= 1:
                continue
            else:
                p = self.fml_joint_pro(fml, f)
                likeli = likeli + log(p + alpha)
        return likeli

    def fml_joint_pro(self, fml, f):
        """
        The joint probability of a family
        """
        parents = fml[:-1]
        kid     = fml[-1]
        X       = np.zeros((1, len(parents)))
        for x, i in zip(parents, range(len(parents))):
            if x < 1: # The first node is the root node
                X[0, 0] = f
            else:
                value = self.tan.parameters.obs_ass[x]
                X[0, i] = value
        y = self.tan.parameters.obs_ass[kid]
        Y = np.array([y])
        beta, var = self.tan.parameters.fml_tank[fml]
        p = obs_pro(X, Y, beta, var)
        return p
