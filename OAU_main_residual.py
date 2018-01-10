"""
this file use residual to detect fault in OAU system
"""
from data_manger.data_oau import DataOAU
import matplotlib.pyplot as pl
import numpy as np
from OAU.OAU_data import get_data
from scipy import stats

file_name = 'OAU/OAU1201.csv'
_, norm_oau_data, n_residuals, labels = get_data(file_name)

alpha = 0.99
x = stats.norm
thresh = x.ppf(1-(1-alpha)/2)

num_f0_r0 = 1
num_f0_r1 = 1
num_f1_r0 = 1
num_f1_r1 = 1
Z_test = []
for r, l in zip(n_residuals, labels):
    f = 0 if l == 0 or l == 1 else 1 #normal, off
    z0 = [i > thresh for i in r]
    z = 0 if sum(z0) == 0 else 1
    Z_test.append(z)
    if f == 0 and z == 0:
        num_f0_r0 = num_f0_r0 + 1
    elif f == 0 and z == 1:
        num_f0_r1 = num_f0_r1 + 1
    elif f == 1 and z == 0:
        num_f1_r0 = num_f1_r0 + 1
    elif f == 1 and z == 1:
        num_f1_r1 = num_f1_r1 + 1
    print("l={},r={}".format(str(l), str(r)))

P_f0_r0 = num_f0_r0 / (num_f0_r0 + num_f0_r1)
P_f0_r1 = num_f0_r1 / (num_f0_r0 + num_f0_r1)
P_f1_r0 = num_f1_r0 / (num_f1_r0 + num_f1_r1)
P_f1_r1 = num_f1_r1 / (num_f1_r0 + num_f1_r1)

pl.figure(1)
pl.scatter(np.array(range(len(Z_test))), np.array(Z_test))
pl.title("Z-test")
pl.xlabel("Samples")
pl.ylabel("Result")
pl.show()

print('end')
