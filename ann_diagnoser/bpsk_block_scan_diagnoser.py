"""
This file uses block scan method to diagnosis BPSK system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from block_scan_layer import MinBlcokScan
from block_scan_layer import FullScan
from block_scan_pooling import BlockMaxPooling
from math import ceil

class DiagnoerBlockScan(nn.Module):
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self, step_len):
        super(DiagnoerBlockScan, self).__init__()
        #mini block scanner layer
        data_size = [5, step_len]
        window_size = 10
        dim_relation = [[1], [2], [0, 1, 2, 3], [3, 4]]
        kernel_size = [5, 5, 5, 5]
        self.mbs1 = MinBlcokScan(data_size, window_size, dim_relation, kernel_size)

        #pooling layer
        data_len = sum(kernel_size) * step_len
        pooling_size = 10
        pooling2_out = sum(kernel_size) * ceil(step_len / pooling_size)
        self.bsp2 = BlockMaxPooling(data_len, step_len, pooling_size)

        #full connectin layer
        self.fc3 = nn.Linear(pooling2_out, 6)

    def forward(self, x):
        x = F.relu(self.mbs1(x))
        x = self.bsp2(x)
        x = F.sigmoid(self.fc3(x))
        return x