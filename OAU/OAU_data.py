# #############################################################################
# This file is used to process data in OAU1201.csv
#
# #############################################################################
import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

def residual_airflow_balance(Q, delta, k, N):
    """
    air flow balance residual
    """
    return Q - delta*k*N

def residual_pressure_balance(P, delta, k, N):
    """
    pressure balance residual
    """
    return P - delta*k*(N**2)

def get_data(file_name):
    """
    pre-process data in OAU system
    """
    raw_oau_data = pd.read_csv(file_name)

    # #############################################################################
    # delete nan
    raw_oau_data = raw_oau_data.dropna(axis=0, how='any')

    # #############################################################################
    # obtain data
    oau_headings = np.array(raw_oau_data.columns[1:])
    oau_data = np.array(raw_oau_data[oau_headings])

    # #############################################################################
    # convert string to number
    for i in range(12, len(oau_data[0])):
        dis_data = oau_data[:, i]
        domain = list(set(dis_data))
        if "On" in domain:
            domain = ["Off", "On"]
        oau_data[:, i] = [domain.index(k) for k in dis_data]

    norm_oau_data = MinMaxScaler().fit_transform(oau_data)
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.02, min_samples=10).fit(norm_oau_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # #############################################################################
    # residuals
    Qe = oau_data[:, 5]
    Qo = oau_data[:, 4]
    Ne = oau_data[:, 3]
    No = oau_data[:, 2]
    Pe = oau_data[:, 1]
    Po = oau_data[:, 0]
    De = oau_data[:, 13]
    Do = oau_data[:, 12]
    k1 = 94.648833656164527
    k2 = 101.05265049003404
    k3 = 0.00016317278844739733
    k4 = 0.00018009338961702802
    residuals = [[residual_airflow_balance(qe, de, k1, ne),\
                  residual_airflow_balance(qo, do, k2, no),\
                  residual_pressure_balance(pe, de, k3, ne),\
                  residual_pressure_balance(po, do, k4, no)] \
                  for qe, qo, pe, po, ne, no, de, do in zip(Qe, Qo, Pe, Po, Ne, No, De, Do)]
    residuals = np.array(residuals)

    normal_residuals = [i for i, j in zip(residuals, labels) if j == 0]
    mean = np.mean(normal_residuals, 0)
    var = np.var(normal_residuals, 0)
    n_residuals = abs((residuals - mean)/(var**0.5))

    return oau_data, norm_oau_data, n_residuals, labels

if __name__ == '__main__':
    _, _, _, labels = get_data('OAU1201.csv')
    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    plt.scatter(x, labels)

    plt.title('Estimated number of clusters: %d' % len(set(labels)))
    plt.show()
