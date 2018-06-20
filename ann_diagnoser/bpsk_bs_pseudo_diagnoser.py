"""
This file uses block scan method to extract features from BPSK system
and then combine the residuals and conduct a hybrid diagnosis
"""

import torch
import torch.nn as nn

class BlockScanPD(nn.Module):#Pseudo Diagnoser
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self):#step_len=100
        super(BlockScanPD, self).__init__()
        #feature extract
        window = 5
        #residual 0
        self.re0_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )

        self.merge_sequence = nn.Sequential(
                            nn.Conv1d(20, 10, 1),
                            nn.ReLU(),
                        )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(10*20, 32),
                            nn.ReLU(),
                            nn.BatchNorm1d(32),
                            nn.Linear(32, 3),
                            nn.Sigmoid(),
                          )

    def forward(self, x):
        x = self.re0_sequence(x)
        x = self.merge_sequence(x)
        x = x.view(-1, 10*20)
        x = self.fc_sequence(x)
        return x
