"""
Search for one or multiple diagnoses based on TAN classifier
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import pickle
import time
import numpy as np
from hybrid_algorithm.a_star_frame import a_star_frame
from hybrid_algorithm.hybrid_tan_classifier import tan_classifier
from math import log

class hybrid_tan_diagnoser(a_star_frame):
    """
    A* search here
    """
    def __init__(self):
        super(hybrid_tan_diagnoser, self).__init__()
        #tan list
        self.tan                   = []
        for _ in range(6):
            tan = tan_classifier()
            self.tan.append(tan)
        #priori knowledge
        self.priori_knowledge      = None
        #obs
        self.obs                   = None

    #interface
    def load_file(self, file_prefix, flag):
        """
        load model files into tan classifiers
        """
        for i in range(6):
            file = file_prefix + str(i) + flag + ".bn"
            with open(file, "rb") as f:
                tan = f.read()
                model = pickle.loads(tan)
                self.tan[i].set_tan(model)

    def set_priori_knowledge(self, priori):
        """
        RT
        """
        self.priori_knowledge = priori

    def set_priori(self, priori):
        """
        set the priori and add them into self.tan
        """
        self.priori = list(priori)
        #add into self.tan
        for i in range(6):
            self.tan[i].set_priori(self.priori[i])

    def update_priori(self):
        """
        update priori probabilities based on the observe
        """
        start = time.clock()
        for i in range(6):
            p0 = self.tan[i].normal_pro()
            p1 = 1 - p0
            self.priori[i] = (p0, p1)
        end = time.clock()
        self.time_stats = self.time_stats + (end - start)

    def add_obs(self, obs_list):
        """
        add the obs list
        """
        self.obs  = np.array(obs_list)
        for i in range(6):
            kids_index      = (self.priori_knowledge[i, 6:]==1)
            kids_obs_list   = self.obs[kids_index]
            n = len(kids_obs_list)
            kids_obs_list   = [(i+1, obs[1]) for i, obs in zip(range(n), kids_obs_list)]
            self.tan[i].add_obs(kids_obs_list)

    #inner function
    def _cost(self, candidate):
        """
        compute the cost of candidate
        """
        #to avoid nummerci problems
        alpha = 1e-20
        #initial cost
        cost = 0
        #priori cost
        for c, i in zip(candidate, range(len(candidate))):
            id = self.order[i]
            value = c
            p_c = self.priori[id][value]
            #to avoid numeric problems
            p_c = (p_c + alpha) /(1 + alpha)
            cost_c = -log(p_c)
            cost = cost + cost_c
        return cost
