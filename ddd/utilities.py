"""
some utilities
"""
import numpy as np
import matplotlib.pyplot as pl
from graph_model.utilities import vector2number as vec2num

def first_one(X):
    """
    find the first one in x
    """
    for x, i in zip(X, range(len(X))):
        if x == 1:
            return (i + 1)
    return 0

def hist_batch(batch_data):
    """
    scatter the batch in different modes
    """
    lb = np.array([first_one(x) for x in batch_data[:, :6]])
    n = len(batch_data[0,:]) - 6
    for k in range(n):
        sk = batch_data[:, k+6]
        pl.figure(k+1)
        for i in range(6):
            mask = (lb==i)
            sk_i = sk[mask]
            pl.subplot(6,1,i+1)
            pl.hist(sk_i, 30)
    pl.show()

def scatter_batch(batch_data):
    """
    scatter batch
    """
    n = len(batch_data[0, :]) - 6
    plt = batch_data[:, 6:]
    lb = np.array([first_one(x) for x in batch_data[:, :6]])
    x = round(np.sqrt(n))
    y = n//x + (0 if n%9 == 0 else 1)
    pl.figure()
    for i in range(n):
        pl.subplot(x, y, i+1)
        pl.scatter(lb, plt[:, i])
    pl.show()

def organise_data(inputs, labels, res, feature):
    """
    get all features, res0, res1 and res2
    """
    length = len(feature)
    fea_num = len(feature[0,:])
    batch_data = np.zeros((length, 6+fea_num+3))
    #the first 6 colums(0:6) are fault labels
    #["tma", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify", "tmb"]
    fault_labels = labels.detach().numpy()
    batch_data[:,:6] = fault_labels
    #the mid fea_num colums(6:6+fea_num) are features
    feature = feature.detach().numpy()
    batch_data[:, 6:6+fea_num] = feature
    #the last 3 colums(6+fea_num:) are residuals
    res = np.array(res)
    #res12
    res = np.mean(np.abs(res), axis=2)
    batch_data[:,-3:-1] = res[:, :2]
    #res3
    inputs = inputs.detach().numpy()
    # s3 = np.mean(inputs[:, 3], axis=1)
    # s4 = np.mean(inputs[:, 4], axis=1)
    # batch_data[:, -1] = ( s4 - 10 * s3)
    s3 = inputs[:, 3]
    s4 = inputs[:, 4]
    batch_data[:, -1] = np.mean(np.abs( s4 - 10 * s3), axis=1)
    return batch_data
