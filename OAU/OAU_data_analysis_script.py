# #############################################################################
# This file is used to analyse data in OAU1201.csv
#
# #############################################################################
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

#residual formula
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


#analyse data in OAU system
file_name = "OAU1201.csv"
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
k1 = 86.2834
k2 = 91.84286
k3 = 0.0001472726
k4 = 0.0001595658
residuals = [[residual_airflow_balance(qe, de, k1, ne),\
                residual_airflow_balance(qo, do, k2, no),\
                residual_pressure_balance(pe, de, k3, ne),\
                residual_pressure_balance(po, do, k4, no)] \
                for qe, qo, pe, po, ne, no, de, do in zip(Qe, Qo, Pe, Po, Ne, No, De, Do)]
residuals = np.array(residuals)

#add data into different list based on the cluaters
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []

#TODO

plt.figure(1)
x = np.arange(len(labels))
plt.scatter(x, labels)
plt.title('Estimated number of clusters: %d' % len(set(labels)))

plt.figure(2)
plt.scatter(x, residuals[:,0])
plt.title('residual 1')

plt.figure(3)
plt.scatter(x,residuals[:,1])
plt.title('residual 2')

plt.figure(4)
plt.scatter(x, residuals[:,2])
plt.title('residual 3')

plt.figure(5)
plt.scatter(x, residuals[:,3])
plt.title('residual 4')
plt.show()

print("end")