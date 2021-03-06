"""
The Hybrid diagnoser combing ANN and 1-SVM
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from scipy import stats
from hybrid_algorithm.a_star_frame import a_star_frame
from hybrid_algorithm.utilities import hypothesis_test as obs_pro
from math import log

class ann1svm_diagnoser(a_star_frame):
    """
    search diagnosis using 1svm as the likelihood estimator
    """
    def __init__(self, p0=0.99):
        super(ann1svm_diagnoser, self).__init__()
        self.obs    = None
        self.svm    = None
        self.p0     = p0

    def load_svm(self, file):
        """
        RT
        """
        self.svm    = joblib.load(file)

    def add_obs(self, obs):
        """
        add obs
        """
        self.obs    = obs

    def _cost(self, candidate):
        """
        compute the cost
        """
        alpha = 1e-20
        p0    = self.p0
        cost  = 0
        for c, i in zip(candidate, range(len(candidate))):
            id = self.order[i]
            value = c
            p_c = self.priori[id][value]
            p_c = (p_c + alpha) /(1 + alpha)
            cost_c = -log(p_c)
            cost = cost + cost_c
        if len(candidate) == 6:
            inputs = np.concatenate((candidate, self.obs), 0)
            inputs = inputs.reshape(1, len(inputs))
            if (self.svm.predict(inputs) == 1).all():
                cost = cost - log(p0)
            else:
                cost = cost - log(1-p0)
        return cost
