"""
some utilities
"""
import numpy as np
from scipy import stats

def hypothesis_test(X, Y, beta, var, alpha=0.95):
    """
    Gaussian hypothesis test
    """
    i_f = 0.2
    var_basis = 1e-4
    N = len(Y)
    e = np.ones((N, 1))
    X = np.hstack((e, X))
    X = np.mat(X)
    mean = X * beta
    mean = np.array(mean)
    var = np.abs(X * var) + var_basis
    var = np.array(var)
    std  = np.sqrt(var)
    interval=stats.norm.interval(alpha,mean,std)
    down = interval[0] - i_f * np.abs(interval[0])
    up   = interval[1] + i_f * np.abs(interval[1])
    
    cmp = ( down < Y) * (Y < up)
    positive = np.sum(cmp)
    negative = N - positive
    p = (positive * alpha + negative * (1-alpha))/N
    return p

def priori_vec2tup(priori):
    """
    convert a priori from vector to tuple
    """
    tup_pri = []
    for p in priori:
        tup_pri.append((1-p, p))
    tup_pri = tuple(tup_pri)
    return tup_pri
