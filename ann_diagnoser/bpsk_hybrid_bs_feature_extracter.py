"""
This file uses block scan method to extract features from BPSK system
and then combine the residuals and conduct a hybrid diagnosis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class HBlockScanFE(nn.Module):#Hybrid Diagnoser, HD
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self):#step_len=100
        super(HBlockScanFE, self).__init__()
        #feature extract
        window = 5
        #based on the influence graph: dim_relation = [[1], [2], [1, 2, 3], [3, 4]]
        #pesudo
        self.fe0_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                            nn.Conv1d(20, 1, 1),
                            nn.AvgPool1d(20),
                          )
        #carrier
        self.fe1_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                            nn.Conv1d(20, 1, 1),
                            nn.AvgPool1d(20),
                          )
        #mixer  now, in influence graph mode. 4-->3 in nn.Conv1d(4, 10, window, padding=window//2)
        self.fe2_sequence = nn.Sequential(
                            nn.Conv1d(3, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                            nn.Conv1d(20, 1, 1),
                            nn.AvgPool1d(20),
                          )
        #amplifier
        self.fe3_sequence = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                            nn.Conv1d(20, 1, 1),
                            nn.AvgPool1d(20),
                          )

        #residual 0
        self.re0_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                            nn.Conv1d(20, 1, 1),
                            nn.AvgPool1d(20),
                          )

        #residual 1
        self.re1_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                            nn.Conv1d(20, 1, 1),
                            nn.AvgPool1d(20),
                          )

        #residual 2
        self.re2_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                            nn.Conv1d(20, 1, 1),
                            nn.AvgPool1d(20),
                          )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(7, 7),
                            nn.ReLU(),
                            nn.BatchNorm1d(7),
                            nn.Linear(7, 6),
                            nn.Sigmoid(),
                          )

    def fe(self, x):
        x0 = x[:, [1], :]               #p
        x1 = x[:, [2], :]               #c
        x2 = x[:, [1, 2, 3], :]         #m      now, in influence graph mode. [0, 1, 2, 3] --> [1, 2, 3]
        x3 = x[:, [3, 4], :]            #a
        r0 = x[:, [5], :]               #r0
        r1 = x[:, [6], :]               #r1
        r2 = x[:, [7], :]               #2
        x0 = self.fe0_sequence(x0)
        x1 = self.fe1_sequence(x1)
        x2 = self.fe2_sequence(x2)
        x3 = self.fe3_sequence(x3)
        r0 = self.re0_sequence(r0)
        r1 = self.re1_sequence(r1)
        r2 = self.re2_sequence(r2)
        x = torch.cat((x0, x1, x2, x3, r0, r1, r2), 1)
        x = x.view(-1, 7)
        return x

    def fp(self, x):
        x = self.fc_sequence(x)
        return x

    def forward(self, x):
        x = self.fe(x)
        x = self.fp(x)
        return x
