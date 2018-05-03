"""
learning structure from data
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir) 
from data_manger.utilities import get_file_list
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd


#prepare data
DATA_PATH = parentdir + "\\bpsk_navigate\\data\\test\\"
list_files = get_file_list(DATA_PATH)
graph = 0
for i, file in enumerate(list_files):
    data = np.load(DATA_PATH + file)
    split = int(len(data) / 2)
    normal = data[:split,:]
    fault = data[split:,:]
    df0 = pd.DataFrame(normal)
    df1 = pd.DataFrame(fault)
    tmp0 = df0.corr('spearman')
    tmp1 = df1.corr('spearman')
    tmp = (tmp0 + tmp1 )/2
    graph = (graph * (i/(i + 1))) + tmp / (i+1)

print(graph)
