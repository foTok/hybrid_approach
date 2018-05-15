"""
some utilities
"""

import numpy as np

def number2vector(num, n):
    """
    convert a int number into a n bit vector
    """
    str_vec = bin(num).replace('0b','').zfill(n)
    vec = np.array([int(i) for i in str_vec])
    return vec

def vector2number(vec):
    """
    convert a vector to number
    """
    str_vec = [str(int(i)) for i in vec]
    str_bin_num = ''.join(str_vec)
    num = int(str_bin_num, 2)
    return num

def organise_data(inputs, labels, res, feature):
    """
    This function only works for BPSK system with 12 features
    where 2 are insignificant
    """
    length = len(feature)
    batch_data = np.zeros((length, 6+12+3))
    #the first 6 colums(0:6) are fault labels
    #["tma", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify", "tmb"]
    fault_labels = labels.detach().numpy()
    batch_data[:,:6] = fault_labels
    #the mid 12 colums(6:18) are features
    feature = feature.detach().numpy()
    batch_data[:, 6:18] = feature
    #the last 3 colums(18:21) are residuals
    res = np.array(res)
    #res12
    res = np.mean(np.abs(res), axis=2)
    batch_data[:,18:20] = res[:, :2]
    #res3
    inputs = inputs.detach().numpy()
    # s3 = np.mean(inputs[:, 3], axis=1)
    # s4 = np.mean(inputs[:, 4], axis=1)
    # batch_data[:, -1] = ( s4 - 10 * s3)
    s3 = inputs[:, 3]
    s4 = inputs[:, 4]
    batch_data[:, -1] = np.mean(np.abs( s4 - 10 * s3), axis=1)
    # return batch_data

    a1 = batch_data[:, :7]
    a2 = batch_data[:, 8:17]
    a3 = batch_data[:, 18:]
    real_data = np.concatenate((a1,a2,a3), axis=1)
    
    return real_data
