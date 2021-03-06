"""
This file uses block scan method to extract features from BPSK system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BlockScanFE(nn.Module):#feature extracter, FE
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self):#step_len=100
        super(BlockScanFE, self).__init__()
        #feature extract
        window = 5
        #based on physical connection: dim_relation = [[1], [2], [0, 1, 2, 3], [3, 4]]
        #based on the influence graph: dim_relation = [[1], [2], [1, 2, 3], [3, 4]]
        #pesudo
        self.fe0_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #carrier
        self.fe1_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #mixer  now, in influence graph mode. 4-->3 in nn.Conv1d(4, 10, window, padding=window//2)
        self.fe2_sequence = nn.Sequential(
                            nn.Conv1d(3, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #amplifier
        self.fe3_sequence = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )

        self.merge_sequence = nn.Sequential(
                            nn.Conv1d(4*20, 40, 1),
                            nn.ReLU(),
                        )

        self.fc0_sequence = nn.Sequential(
                            nn.Linear(40*20, 12),
                            nn.ReLU(),
                            nn.BatchNorm1d(12),
                          )
        
        #fault predictor
        self.fc1 = nn.Sequential(
                            nn.Linear(12, 6),
                            nn.Sigmoid(),
                          )
    def fe(self, x):
        x0 = x[:, [1], :]               #p
        x1 = x[:, [2], :]               #c
        x2 = x[:, [1, 2, 3], :]         #m      now, in influence graph mode. [0, 1, 2, 3] --> [1, 2, 3]
        x3 = x[:, [3, 4], :]            #a
        x0 = self.fe0_sequence(x0)
        x1 = self.fe1_sequence(x1)
        x2 = self.fe2_sequence(x2)
        x3 = self.fe3_sequence(x3)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = x.view(-1, 4*20, 20)
        x = self.merge_sequence(x)
        x = x.view(-1, 40*20)
        x = self.fc0_sequence(x)
        return x

    def fp(self, x):
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.fe(x)
        x = self.fp(x)
        return x