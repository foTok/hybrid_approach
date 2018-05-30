"""
TAN learning
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from math import log
from graph_model.Bayesian_network import Bayesian_network
from graph_model.graph_component import Bayesian_structure
from graph_model.graph_component import Bayesian_Gaussian_parameter
from graph_model.utilities import min_span_tree

class tan_learning:
    """
    learning a TAN structure
    """
    def __init__(self):
        #the data
        self.batch = None
        #bins 10 by default
        self.bins  = 10
        #fault number fn
        self.fn    = 6
        #fault + feature number fen
        self.ffn   = None
        #hist cache
        #key: int or tuple(small,big).
        #value: numpy.array. 1 or 2 dimensions.
        self.hist_cache = {}
        #cmp cache
        self.bin_cache = {}

    def set_batch(self, batch):
        """
        RT
        """
        self.batch = batch
        #feature number
        _, ffn = batch.shape
        self.ffn  = ffn

    def set_bins(self, bins):
        """
        RT
        """
        self.bins = bins

    def set_fn(self, fn):
        """
        RT
        """
        self.fn = fn

    def learn_mst(self):
        """
        learn a minimal span tree
        """
        MIEM = self.learn_feature_MIE()
        mst = min_span_tree(MIEM)
        return mst

    def learn_feature_MIE(self):
        """
        RT
        """
        #feature number
        fen = self.ffn - self.fn
        #Mutual Information Entropy Matrix
        MIEM = np.zeros((fen, fen))
        for i in range(self.fn, self.ffn):
            for j in range(i+1, self.ffn):
                MIE = self.learn_feature_MIEij(i, j)
                MIEM[i-self.fn, j-self.fn] = MIE
                MIEM[j-self.fn, i-self.fn] = MIE
        return MIEM

    def learn_feature_MIEij(self, i, j):
        """
        RT
        """
        assert i < j
        histi = self.hist(i)
        histj = self.hist(j)
        histij = self.histij(i, j)
        MIE = 0
        for i0 in range(self.bins):
            for j0 in range(self.bins):
                relative_pro = histij[i0, j0] / (histi[i0] * histj[j0])
                MIE0 = histij[i0, j0] * log(relative_pro)
                MIE  = MIE + MIE0
        return MIE

    def hist(self, i):
        """
        learn the parameters of discritized feature i
        """
        assert 0 <= i <self.ffn
        if i in self.hist_cache:
            hist = self.hist_cache[i]
        else:
            hist = self.hist_from_batch(i)
            self.hist_cache[i] = hist
        return hist

    def histij(self, i, j):
        """
        learn the discritized features i, j
        """
        assert 0 <= i < j <self.ffn
        if (i, j) in self.hist_cache:
            hist = self.hist_cache[(i, j)]
        else:
            hist = self.histij_from_batch(i, j)
            self.hist_cache[(i, j)] = hist
        return hist

    def hist_from_batch(self, i):
        """
        RT
        """
        hist  = np.zeros((self.bins, 1))
        batch = self.batch[:, i]
        max_value = np.max(batch)
        min_value = np.min(batch)
        interval = (max_value - min_value) / self.bins
        assert interval > 0
        for i in range(self.bins):
            hist[i] = np.sum(self.bin_count(batch, min_value, interval, i)) + 1
        #normalization
        hist = hist / (len(self.batch) + self.bins)
        return hist
                
    
    def histij_from_batch(self, i, j):
        """
        RT
        """
        assert 0 <= i < j <self.ffn
        hist  = np.zeros((self.bins, self.bins))
        #data i
        batch_i = self.batch[:, i]
        max_i = np.max(batch_i)
        min_i = np.min(batch_i)
        int_i = (max_i - min_i) / self.bins
        #data j
        batch_j = self.batch[:, j]
        max_j = np.max(batch_j)
        min_j = np.min(batch_j)
        int_j = (max_j - min_j) / self.bins
        #count start
        for i in range(self.bins):
            for j in range(self.bins):
                hist_i = self.bin_count(batch_i, min_i, int_i, i)
                hist_j = self.bin_count(batch_j, min_j, int_j, j)
                hist_ij = hist_i * hist_j
                hist[i, j] = np.sum(hist_ij) + 1
        #normalization
        hist = hist / (self.batch.size + self.bins**2)
        return hist
    
    def bin_count(self, batch, min_value, interval, i):
        """
        return if the batch in the i_th bins
        """
        if i in self.bin_cache:
            hist = self.bin_cache[i]
        else:
            hist = self.bin_count_from_batch(batch, min_value, interval, i)
            self.bin_cache[i] = hist
        return hist

    def bin_count_from_batch(self, batch, min_value, interval, i):
        """
        return if the batch in the i_th bins from batch
        """
        assert 0 <= i <self.bins
        if i == 0:#first bin
            hist = (batch < (min_value + interval))
        elif i == (self.bins - 1): #last bin
            hist = ((min_value + i*interval) <= batch)
        else:#other bins
            lower_bound = min_value + i * interval
            upper_bound = min_value + (i+1) * interval
            hist = ((lower_bound <= batch) * (batch < upper_bound))
        return hist
