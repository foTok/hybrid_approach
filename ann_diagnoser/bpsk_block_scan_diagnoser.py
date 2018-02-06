"""
This file uses block scan method to diagnosis BPSK system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from block_scan_layer import MinBlcokScan
from block_scan_layer import FullScan

class DiagnoerBlockScan(nn.Module):
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self, step_len):
        super(DiagnoerBlockScan, self).__init__()
        self.fc1 = nn.Linear(step_len, 4*step_len)
        self.fc2 = nn.Linear(4*step_len, 2*step_len)
        self.fc3 = nn.Linear(2*step_len, step_len)
        self.fc4 = nn.Linear(step_len, int(step_len / 2))
        self.fc5 = nn.Linear(int(step_len / 2), 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x