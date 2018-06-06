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

#settings
PATH = parentdir
DATA_PATH = PATH + "\\bpsk_navigate\\data\\test\\"
ANN_PATH = PATH + "\\ddd\\ann_model\\"
ann_file = "FE0.pkl"

#prepare data
mana = BpskDataTank()

step_len=100
list_files = get_file_list(DATA_PATH)
for file in list_files:
    mana.read_data(DATA_PATH+file, step_len=step_len, snr=20)

batch = 20000
FE_test = torch.load(ANN_PATH + ann_file)
FE_test.eval()
inputs, labels, _, _ = mana.random_batch(batch, normal=0.2, single_fault=10, two_fault=1)
features = FE_test.fe(inputs)
features = features.detach().numpy()

labels = labels.numpy()

#color
color = [vector2number(x) for x in labels]
color = np.array(color)

F_embedded = TSNE(n_components=2).fit_transform(features)
plt.scatter(F_embedded[:, 0], F_embedded[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("t-SNE")
plt.show()
