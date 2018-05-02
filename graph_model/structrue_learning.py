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
for file in list_files:
    #TODO
    #read all data.
    #store it in pandas and analyze it.
data = pd.DataFrame()
