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
        self.fe_p         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #residual 0
        self.fe_r0         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #carrier
        self.fe_c          = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #residual 1
        self.fe_r1         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #mixer
        self.fe_m          = nn.Sequential(
                            nn.Conv1d(5, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #amplifier
        self.fe_a          = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #residual 2
        self.fe_r2         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window),
                          )
        #information merger
        self.merge_sequence = nn.Sequential(
                            nn.Conv1d(20*7, 40, 1),
                            nn.ReLU(),
                        )
        #fault predictor
        self.fc_sequence = nn.Sequential(
                            nn.Linear(40*20, 128),
                            nn.ReLU(),
                            nn.BatchNorm1d(128),
                            nn.Linear(128, 6),
                            nn.Sigmoid(),
                          )

    def forward(self, x):
        x0 = x[:, [1], :]               #p
        x1 = x[:, [2], :]               #c
        x2 = x[:, [1, 2, 3, 5, 6], :]   #p, c, m, r0, r1
        x3 = x[:, [3, 4], :]            #m, a, r2
        r0 = x[:, [5], :]               #r0
        r1 = x[:, [6], :]               #r1
        r2 = x[:, [7], :]               #r2
        x0 = self.fe_p(x0)
        x1 = self.fe_c(x1)
        x2 = self.fe_m(x2)
        x3 = self.fe_a(x3)
        r0 = self.fe_r0(r0)
        r1 = self.fe_r1(r1)
        r2 = self.fe_r2(r2)
        x = torch.cat((x0, x1, x2, x3, r0, r1, r2), 1)
        x = x.view(-1, 20*7, 20)
        x = self.merge_sequence(x)
        x = x.view(-1, 40*20)
        x = self.fc_sequence(x)
        return x
