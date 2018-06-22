"""
hybrid influence graph based CNN diagnoser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class higcnn_diagnoser(nn.Module):
    """
    hybrid influence graph based CNN diagnoser
    """
    def __init__(self):#step_len=100
        super(higcnn_diagnoser, self).__init__()
        #feature extract
        window = 5
        #based on the influence graph: dim_relation = [[1], [2], [1, 2, 3], [3, 4]]
        #pesudo
        self.fe0_sequence = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #carrier
        self.fe1_sequence = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #mixer
        self.fe2_sequence = nn.Sequential(
                            nn.Conv1d(5, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #amplifier
        self.fe3_sequence = nn.Sequential(
                            nn.Conv1d(3, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )

        self.merge_sequence = nn.Sequential(
                            nn.Conv1d(20*4, 40, 1),
                            nn.ReLU(),
                        )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(40*20, 128),
                            nn.ReLU(),
                            nn.BatchNorm1d(128),
                            nn.Linear(128, 6),
                            nn.Sigmoid(),
                          )

    def forward(self, x):
        x0 = x[:, [1, 5], :]            #p, r0
        x1 = x[:, [2, 6], :]            #c, r1
        x2 = x[:, [1, 2, 3, 5, 6], :]   #p, c, m, r0, r1
        x3 = x[:, [3, 4, 7], :]         #m, a, r2
        x0 = self.fe0_sequence(x0)
        x1 = self.fe1_sequence(x1)
        x2 = self.fe2_sequence(x2)
        x3 = self.fe3_sequence(x3)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = x.view(-1, 20*4, 20)
        x = self.merge_sequence(x)
        x = x.view(-1, 40*20)
        x = self.fc_sequence(x)
        return x
