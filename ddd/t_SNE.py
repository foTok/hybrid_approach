""""
use tSNE to evaluate featrues
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import torch
from sklearn.manifold import TSNE
import os
from ann_diagnoser.bpsk_block_scan_feature_extracter import BlockScanFE
from data_manger.bpsk_data_tank import BpskDataTank
from data_manger.utilities import get_file_list
import matplotlib.pyplot as plt
from graph_model.utilities import vector2number
from ddd.utilities import organise_tensor_data

#data amount
small_data = True
#settings
obj         = "hfe"  #fe, hfe
PATH        = parentdir
DATA_PATH   = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH    = PATH + "\\ddd\\ann_model\\" + ("big_data\\" if not small_data else "small_data\\")
step_len    = 100
batch       = 2000
norm        = False
if obj == "fe":
    model_file = "FE.pkl"
elif obj == "hfe":
    model_file = "HFE.pkl"
    norm       = True
else:
    print("unkown object!")
    exit(0)

#prepare data
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20, norm=norm)

FE_test = torch.load(ANN_PATH + model_file)
FE_test.eval()
inputs, labels, _, res = mana.random_batch(batch, normal=0.14, single_fault=10, two_fault=0)
sen_res = organise_tensor_data(inputs, res)
features = FE_test.fe(sen_res)
features = features.detach().numpy()

labels = labels.numpy()

#color
color = [vector2number(x) for x in labels]
color = 10*np.array(color) + 10

F_embedded = TSNE(n_components=2).fit_transform(features)
plt.scatter(F_embedded[:, 0], F_embedded[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("t-SNE")
plt.show()
