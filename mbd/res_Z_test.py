"""
Z test for residuals
"""
import numpy as np
from scipy import stats
from numpy import array

def Z_test(res, alpha):
    """
    reture the results of Z-test
    """
    #confidence
    thresh = stats.norm.ppf(1-(1-alpha)/2)
    delta = array([0.01, 0.005, 0.005, 0.3])
    res = array(res)
    res = np.abs(res)
    res = np.mean(res, axis=2)
    weight = delta**0.5
    res = res / weight

    return res > thresh
