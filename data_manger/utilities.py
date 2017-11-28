"""
this file reads data generated by bpsk_navigate
and transfers them so that could be used by diagnoser
"""

import numpy as np

def read_data(file_name, step_len=None, split_point=None):
    """
    read data, and re-organise them
    """
    sig = np.load(file_name)
    split_point = (len(sig) / 2) if split_point is None else split_point
    step_len = 100 if step_len is None else step_len
    #normal data
    normal = []
    for i in range(split_point - step_len):
        normal.append(sig[i:i+step_len, 2:])
    #fault data
    fault = []
    for i in range(split_point, len(sig)-step_len):
        fault.append(sig[i:i+step_len, 2:])

    return normal, fault
