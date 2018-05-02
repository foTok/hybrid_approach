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
        #dim_relation = [[1], [2], [0, 1, 2, 3], [3, 4]]
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
        #mixer
        self.fe2_sequence = nn.Sequential(
                            nn.Conv1d(4, 10, window, padding=window//2),
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

        self.fc0_sequence = nn.Sequential(
                            nn.Linear(4*20*20, 4*64),
                            nn.ReLU(),
                            nn.Linear(4*64, 4*3),
                            nn.ReLU(),
                            nn.BatchNorm1d(4*3),
                          )
        
        #fault predictor
        self.fc1 = nn.Sequential(
                            nn.Linear(4*3, 10),
                            nn.ReLU(),
                            nn.Linear(10, 6),
                            nn.Sigmoid(),
                          )
    def fe(self, x):
        x0 = x[:, [1], :]               #p
        x1 = x[:, [2], :]               #c
        x2 = x[:, [0, 1, 2, 3], :]      #m
        x3 = x[:, [3, 4], :]            #a
        x0 = self.fe0_sequence(x0)
        x1 = self.fe1_sequence(x1)
        x2 = self.fe2_sequence(x2)
        x3 = self.fe3_sequence(x3)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = x.view(-1, 4*20*20)
        x = self.fc0_sequence(x)
        return x

    def fp(self, x):
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.fe(x)
        x = self.fp(x)
        return x