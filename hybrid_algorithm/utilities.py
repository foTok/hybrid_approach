"""
some utilities
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def hypothesis_test(X, Y, beta, var, alpha=0.99):
    """
    Gaussian hypothesis test
    """
    i_f = 0.0
    var_basis = 1e-4
    N = len(Y)
    e = np.ones((N, 1))
    X = np.hstack((e, X))
    X = np.mat(X)
    mean = X * beta
    mean = np.array(mean)
    if isinstance(var, float) or isinstance(var, int):
        var = var
    else:
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

def plot_bar(y, label, std=None, xlabel=None, ylabel=None, xticks=None):
    """
    plot data in a bar
    """
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            height = np.round(height, 3)
            w = rect.get_x()+rect.get_width()/2.- (0.4 if std is not None else 0.2)
            h = (1.03 if std is not None else 1.01)*height
            plt.text(w, h, '%s' % float(height))
    n, size = y.shape
    total_width = 0.8
    width = total_width / n
    x0 = np.arange(size)
    x = x0 - (total_width - width) / 2
    for i, yi, li in zip(range(n), y, label):
        bar = plt.bar(x + width*i, yi, width=width, label=li)
        autolabel(bar)
        if std is not None:
            stdi = std[i]
            plt.errorbar(x + width*i, yi, yerr=stdi*3, ls='none', ecolor='k')
    plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(x0, xticks)
    plt.show()
