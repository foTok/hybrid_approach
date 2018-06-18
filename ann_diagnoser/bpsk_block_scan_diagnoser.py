"""
This file uses block scan method to diagnosis BPSK system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BlockScanDiagnoser(nn.Module):
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self):
        super(BlockScanDiagnoser, self).__init__()
        window = 5
        #based on physical connection: dim_relation = [[1], [2], [0, 1, 2, 3], [3, 4]]
        #based on the influence graph: dim_relation = [[1], [2], [1, 2, 3], [3, 4]]    << currently
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
                            nn.Conv1d(3, 10, window, padding=window//2),
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

        self.merge_sequence = nn.Sequential(
                            nn.Conv1d(4*20, 40, 1),
                            nn.ReLU(),
                        )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(40*20, 80),
                            nn.ReLU(),
                            nn.BatchNorm1d(80),
                            nn.Linear(80, 6),
                            nn.Sigmoid(),
                          )

    def forward(self, x):
        x0 = x[:, [1], :]
        x1 = x[:, [2], :]
        x2 = x[:, [1, 2, 3], :]
        x3 = x[:, [3, 4], :]
        x0 = self.p_sequence(x0)
        x1 = self.c_sequence(x1)
        x2 = self.m_sequence(x2)
        x3 = self.a_sequence(x3)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = x.view(-1, 4*20, 20)
        x = self.merge_sequence(x)
        x = x.view(-1, 40*20)
        x = self.fc_sequence(x)
        return x
