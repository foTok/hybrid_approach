"""
This file uses block scan method to diagnosis BPSK system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from block_scan_pooling import BlockMaxPooling
from math import ceil

class DiagnoerBlockScan(nn.Module):
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self, step_len):
        super(DiagnoerBlockScan, self).__init__()
        window = 5
        #dim_relation = [[1], [2], [0, 1, 2, 3], [3, 4]]
        self.p_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.c_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.m_sequence = nn.Sequential(
                            nn.Conv1d(4, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.a_sequence = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(4*20*20, 4*64),
                            nn.ReLU(),
                            nn.Linear(4*64, 64),
                            nn.ReLU(),
                            nn.Linear(64, 6),
                            nn.Sigmoid()
                          )

    def forward(self, x):
        x1 = x[:, [1], :]
        x2 = x[:, [2], :]
        x3 = x[:, [0, 1, 2, 3], :]
        x4 = x[:, [3, 4], :]
        x1 = self.p_sequence(x1)
        x2 = self.c_sequence(x2)
        x3 = self.m_sequence(x3)
        x4 = self.a_sequence(x4)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = x.view(-1, 4*20*20)
        x = self.fc_sequence(x)
        return x