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

def graphviz_Bayes(struct, file, fea_num = 12):
    """
    convert struct into graphviz file
    """
    labels = ["F[0..6]"]
    for i in range(fea_num):
        labels.append("fe"+str(i))
    labels = labels + ["r0", "r1", "r2"]
    G = Digraph()
    #add nodes
    for i, lab in zip(range(len(labels)), labels):
        if i == 0:
            color = "yellow"
        elif  1<=i<1+fea_num:
            color = "green"
        else:
            color = "red"
        G.node(lab, lab, fillcolor=color, style="filled")
    #add edges from fault to features
    for i in range(fea_num):
        G.edge(labels[0], labels[i+1])
    #add edges from fault to residuals
    G.edge(labels[0], labels[-3], label="[0,1,5]")
    G.edge(labels[0], labels[-2], label="[2]")
    G.edge(labels[0], labels[-1], label="[4]")

    #add edges in struct
    n = len(struct)
    for i in range(6, n):
        for j in range(i+1, n):
            if struct[i, j] == 1:
                G.edge(labels[i-5], labels[j-5])
    print(G)
    G.render(file, view=True)
    print("Saved in: ", file)

def Guassian_cost(batch, fml, beta, var, norm):
    """
    the cost function
    """
    x = fml[:-1]
    y = fml[-1]
    X = batch[:, x]
    Y = batch[:, y]
    cost = Guassian_cost_core(X, Y, beta, var, norm)
    return cost

def Guassian_cost_core(X, Y, beta, var, norm):
    """
    cost function
    """
    var_basis = 1e-4
    N = len(Y)
    e = np.ones((N, 1))
    X = np.hstack((e, X))
    X = np.mat(X)
    Y_p = X * beta
    Y_p.shape = (N,)
    #number
    if isinstance(var, float) or isinstance(var, int):
        Var = var
    else:#vector
        Var = X * var
        Var = np.abs(np.array(Var))+ var_basis
        Var.shape = (N,)

    if norm:
        #cost0 = log(2*pi*var)/2
        cost0 = np.mean(np.log(2*np.pi*Var)) / 2
    else:
        cost0 = 0
    #cost1 = res**2/(2var)    
    cost1 = np.mean(np.array(Y_p - Y)**2 / (2*Var))
    cost = cost0 + cost1
    return cost

def priori_knowledge(fea_num = 12):
    """
    set priori knowledge
    """
    #0 ~can not determine
    #1 ~connection
    #-1~no connection
    #initially, all connection can not be determined.
    n = 6 + fea_num + 3
    pri_knowledge = np.zeros((n, n))
    #no self connection and downstair connection
    for i in range(n):
        for j in range(i+1):
            pri_knowledge[i,j] = -1
    #no connection between faults
    for i in range(6):
        for j in range(i+1, 6):
            pri_knowledge[i, j] = -1
            pri_knowledge[j, i] = -1
    #connections from faults to features or residuals
    #here, we first connect all faults to residuals.
    #but some edges will be deleted.
    for i in range(6):
        for j in range(6,n):
            pri_knowledge[i, j] = 1
    # model information
    #r0 --- node -3
    #unrelated fault [2,3,4]
    uf1 = [2,3,4]
    for i in uf1:
        pri_knowledge[i][-3] = -1
        pri_knowledge[-3][i] = -1
    #r1 --- node -2
    uf2 = [0,1,3,4,5]
    for i in uf2:
        pri_knowledge[i][-2] = -1
        pri_knowledge[-2][i] = -1
    #r3 --- node -1
    uf3 = [0,1,2,3,5]
    for i in uf3:
        pri_knowledge[i][-1] = -1
        pri_knowledge[-1][i] = -1
    #no connection between residuals
    for i in range(-3, 0):
        for j in range(-3, 0):
            pri_knowledge[i, j] = -1

    return pri_knowledge