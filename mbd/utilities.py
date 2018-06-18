"""
some utilities
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from scipy import stats

def residuals_pca(inputs, res):
    """
    residuals of pesude code, carriar and amplifier
    """
    length              = len(res)
    batch_data          = np.zeros((length, 3))
    res                 = np.array(res)
    #res12
    res                 = np.mean(np.abs(res), axis=2)
    batch_data[:, :2]   = res[:, :2]
    #res3
    inputs              = inputs.detach().numpy()
    s3                  = inputs[:, 3]
    s4                  = inputs[:, 4]
    batch_data[:, -1]   = np.mean(np.abs( s4 - 10 * s3), axis=1)
    return batch_data


def hypothesis_test(res, var, alpha):
    """
    return the conflicts based on res, var and confidence
    """
    mean        = np.zeros((1, len(res)))
    interval    = stats.norm.interval(alpha, mean, np.sqrt(var))
    result      = []
    UP          = interval[1].reshape(3)
    DOWN        = interval[0].reshape(3)
    for r, up, down in zip(res, UP, DOWN):
        if r < up and r > down:
            result.append(True)
        else:
            result.append(False)
    return result

def get_conflicts(results, conflict_table):
    """
    find conflict based on results and conflic table
    """
    conflicts       = []
    for c, pas in zip(conflict_table, results):
        if not pas:
            conflicts.append(c)
    return conflicts
