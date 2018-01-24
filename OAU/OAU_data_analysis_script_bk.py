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

#add data into different list based on the cluaters
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []

for res, lab in zip(residuals, labels):
    if lab == 0:
        list1.append(res)
    elif lab == 1:
        list2.append(res)
    elif lab == 2:
        list3.append(res)
    elif lab == 3:
        list4.append(res)
    elif lab == 4:
        list5.append(res)
    elif lab == 5:
        list6.append(res)
    elif lab == -1:
        list7.append(res)

residuals = np.array(residuals)
normal_residuals = [i for i, j in zip(residuals, labels) if j == 0 or j == 1]
mean = np.mean(normal_residuals, 0)
var = np.var(normal_residuals, 0)
n_residuals = abs((residuals - mean)/(var**0.5))

n_residuals2 = abs((residuals - mean)/(var**0.5))
k = 10
for i in range(k, len(residuals)):
    data = residuals[i-k:i,:]
    mean = np.mean(data, 0)
    var = np.var(data, 0) + 1e-6
    n_residuals2[i] = abs((residuals[i] - mean)/(var**0.5))



#confidence alpha
alpha = 0.99
thresh_hold = st.norm.ppf(1 - (1 - alpha) / 2)

r1 = []
r2 = []
r3 = []
r4 = []
num_f0_r0 = 0
num_f0_r1 = 0
num_f1_r0 = 0
num_f1_r1 = 0
for r, l in zip(n_residuals, labels):
    f = 0 if l == 0 or l == 1 else 1 #normal, off
    z = 1 if r[0] > thresh_hold or r[1] > thresh_hold or r[2] > thresh_hold or r[3] > thresh_hold else 0
    if f == 0 and z == 0:
        num_f0_r0 = num_f0_r0 + 1
    elif f == 0 and z == 1:
        num_f0_r1 = num_f0_r1 + 1
    elif f == 1 and z == 0:
        num_f1_r0 = num_f1_r0 + 1
    elif f == 1 and z == 1:
        num_f1_r1 = num_f1_r1 + 1

    r1.append(1 if r[0] > thresh_hold else 0)
    r2.append(1 if r[1] > thresh_hold else 0)
    r3.append(1 if r[2] > thresh_hold else 0)
    r4.append(1 if r[3] > thresh_hold else 0)
r1_actived = sum(r1)
r2_actived = sum(r2)
r3_actived = sum(r3)
r4_actived = sum(r4)





R1 = []
R2 = []
R3 = []
R4 = []
num_f0_R0 = 0
num_f0_R1 = 0
num_f1_R0 = 0
num_f1_R1 = 0
for r, l in zip(n_residuals2, labels):
    f = 0 if l == 0 or l == 1 else 1 #normal, off
    z = 1 if r[0] > thresh_hold or r[1] > thresh_hold or r[2] > thresh_hold or r[3] > thresh_hold else 0
    if f == 0 and z == 0:
        num_f0_R0 = num_f0_R0 + 1
    elif f == 0 and z == 1:
        num_f0_R1 = num_f0_R1 + 1
    elif f == 1 and z == 0:
        num_f1_R0 = num_f1_R0 + 1
    elif f == 1 and z == 1:
        num_f1_R1 = num_f1_R1 + 1
    R1.append(1 if r[0] > thresh_hold else 0)
    R2.append(1 if r[1] > thresh_hold else 0)
    R3.append(1 if r[2] > thresh_hold else 0)
    R4.append(1 if r[3] > thresh_hold else 0)

R1_actived = sum(R1)
R2_actived = sum(R2)
R3_actived = sum(R3)
R4_actived = sum(R4)


print("num_f0_r0 = %d" % num_f0_r0)
print("num_f0_r1 = %d" % num_f0_r1)
print("num_f1_r0 = %d" % num_f1_r0)
print("num_f1_r1 = %d" % num_f1_r1)


print("num_f0_R0 = %d" % num_f0_R0)
print("num_f0_R1 = %d" % num_f0_R1)
print("num_f1_R0 = %d" % num_f1_R0)
print("num_f1_R1 = %d" % num_f1_R1)

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

plt.figure(6)
plt.scatter(x, r1, marker=".")
plt.title('Z-test for residual 1')

plt.figure(7)
plt.scatter(x, r2, marker=".")
plt.title('Z-test for residual 2')

plt.figure(8)
plt.scatter(x, r3, marker=".")
plt.title('Z-test for residual 3')

plt.figure(9)
plt.scatter(x, r4, marker=".")
plt.title('Z-test for residual 4')

plt.figure(10)
plt.scatter(x, R1, marker=".")
plt.title('Z-test for Residual 1')

plt.figure(11)
plt.scatter(x, R2, marker=".")
plt.title('Z-test for Residual 2')

plt.figure(12)
plt.scatter(x, R3, marker=".")
plt.title('Z-test for Residual 3')

plt.figure(13)
plt.scatter(x, R4, marker=".")
plt.title('Z-test for Residual 4')
plt.show()

print("end")