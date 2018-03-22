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
    #[0.01, 0.005, 0.005, 0.3]
    #[  9.97523941e-05   5.14100362e-05   5.16158812e-05   6.13690207e-03]
    delta = array([3.61122403e-05,   1.90020324e-05,   2.22642168e-05,   8.92350168e-03]) * 1.2
    mean = array([0.08004993,  0.05648239,  0.05651077,  0.61044511])
    res = array(res)
    res = np.abs(res)
    res = np.mean(res, axis=2)
    res = np.abs(res-mean)
    weight = delta**0.5
    res = res / weight

    return res > thresh
