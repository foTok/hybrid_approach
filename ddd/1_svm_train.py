"""
The class that train a 1-SVM likelihood esitmater.
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import torch
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
from ddd.utilities import organise_data

#data amount
small_data      = True
nu              = 0.01   #0.01, 0.05, 0.1
#settings
PATH            = parentdir
DATA_PATH       = PATH + "\\bpsk_navigate\\data\\" + ("big_data\\" if not small_data else "small_data\\")
ANN_PATH        = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
SVM_PATH        = PATH + "\\ddd\\svm_model\\" + ("big_data\\" if not small_data else "small_data\\")
step_len        = 100
fe_name         = "FE.pkl"
model_file      = "likelihood.m"

#load fe
FE = torch.load(ANN_PATH + fe_name)
FE.eval()
#prepare data
mana = BpskDataTank()
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=True)

#train
batch = 20000
train_loss = []
running_loss = 0.0
#sample data
inputs, labels, _, res = mana.random_batch(batch, normal=0.4, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
X_train = organise_data(inputs, labels, res, feature)

#1-SVM Model
clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)
clf.fit(X_train)

#save Model
joblib.dump(clf, SVM_PATH + model_file)

#sample test
inputs, labels, _, res = mana.random_batch(2000, normal=0.4, single_fault=10, two_fault=0)
feature = FE.fe(inputs)
X_test = organise_data(inputs, labels, res, feature)
#evaluate
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
error_train = y_pred_train[y_pred_train == -1].size / len(y_pred_train)
error_test = y_pred_test[y_pred_test == -1].size / len(y_pred_test)

#print
print("error rate train=", error_train)
print("error ratetest=", error_test)
print("DONE")
