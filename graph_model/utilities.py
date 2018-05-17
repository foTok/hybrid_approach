"""
some utilities
"""

import numpy as np
import matplotlib.pyplot as pl
from graphviz import Digraph

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

def hist_batch(batch_data):
    """
    scatter the batch in different modes
    """
    lb = np.array([np.argwhere(x == 1)[0][0] for x in batch_data[:, :6]])
    for k in range(13):
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
    n = len(batch_data[0, :])
    plt = batch_data[:, 6:]
    lb = np.array([np.argwhere(x == 1)[0][0] for x in batch_data[:, :6]])
    pl.figure()
    for i in range(n-6):
        pl.subplot(5, 3, i+1)
        pl.scatter(lb, plt[:, i])
    pl.show()

def graphviz_Bayes(struct, file):
    """
    convert struct into graphviz file
    """
    labels = ["F[0..6]",\
          "fe0", "fe1", "fe2", "fe3", "fe4", "fe5", "fe6", "fe7", "fe8", "fe9",\
          "r1", "r2", "r3"]
    G = Digraph()
    #add nodes
    for i, lab in zip(range(len(labels)), labels):
        if i == 0:
            color = "yellow"
        elif  1<=i<11:
            color = "green"
        else:
            color = "red"
        G.node(lab, lab, fillcolor=color, style="filled")
    #add edges from fault to features
    for i in range(10):
        G.edge(labels[0], labels[i+1])
    #add edges from fault to residuals
    G.edge(labels[0], labels[11], label="[0,1,5]")
    G.edge(labels[0], labels[12], label="[2]")
    G.edge(labels[0], labels[13], label="[4]")

    #add edges in struct
    for i in range(6, 19):
        for j in range(i+1, 19):
            if struct[i, j] == 1:
                G.edge(labels[i-5], labels[j-5])
    print(G)
    G.render(file, view=True)
    print("Saved in: ", file)

def Guassian_cost(batch, fml, beta, var):
    """
    the cost function
    """
    var_basis = 1e-3
    x = fml[:-1]
    y = fml[-1]
    N = len(batch)
    e = np.ones((N, 1))
    X = np.hstack((e, batch[:, x]))
    X = np.mat(X)
    Y = batch[:, y]
    Y_p = X * beta
    Var = X * var
    Y_p.shape = (N,)
    Var.shape = (N,)
    res = Y_p - Y

    res = np.abs(np.array(res))
    Var = np.abs(np.array(Var))+ var_basis

    relative_res = (res**2) / (2*Var)
    cost = np.mean(relative_res)
    return cost
