"""
hybrid influence graph based structrual convolutional neural network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class rdscnn_diagnoser(nn.Module):
    """
    hybrid influence graph based structrual convolutional neural network diagnoser
    """
    def __init__(self):
        super(rdscnn_diagnoser, self).__init__()
        window = 5
        #feature extractor
        # block 0
        self.fe_p         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )
        # block 1
        self.fe_r0        = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )
        # block 2
        self.fe_c         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )
        # block 3
        self.fe_r1        = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )
        # m, p, c, s0
        # block 4
        self.fe_mpcs0     = nn.Sequential(
                            nn.Conv1d(4, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )
        # s0, s1
        # block 5
        self.fe_s01        = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )
        # block 6
        self.fe_r2        = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        # internal connector
        # block 7
        self.ic0         = nn.Sequential(
                            nn.Conv1d(100, 20, 1),
                            nn.ReLU(),
                          )
        # block 8
        self.ic1         = nn.Sequential(
                            nn.Conv1d(60, 20, 1),
                            nn.ReLU(),
                          )
        #information merger
        # block 9
        self.m0          = nn.Sequential(
                            nn.Conv1d(80, 20, 1),
                            nn.ReLU(),
                          )
        # block 10
        self.m1          = nn.Sequential(
                            nn.Conv1d(80, 20, 1),
                            nn.ReLU(),
                          )
        # block 11
        self.m2          = nn.Sequential(
                            nn.Conv1d(80, 20, 1),
                            nn.ReLU(),
                          )
        # block 12
        self.m3          = nn.Sequential(
                            nn.Conv1d(40, 20, 1),
                            nn.ReLU(),
                          )
        # block 13
        self.m4          = nn.Sequential(
                            nn.Conv1d(20, 20, 1),
                            nn.ReLU(),
                          )
        # block 14
        self.m5          = nn.Sequential(
                            nn.Conv1d(80, 20, 1),
                            nn.ReLU(),
                          )
        #fault predictor
        self.fp0         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                            nn.Sigmoid(),
                          )

        self.fp1         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                            nn.Sigmoid(),
                          )

        self.fp2         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                            nn.Sigmoid(),
                          )

        self.fp3         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                            nn.Sigmoid(),
                          )

        self.fp4         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                            nn.Sigmoid(),
                          )

        self.fp5         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                            nn.Sigmoid(),
                          )

    def forward(self, x):
        #extract family
        #m, p, c, s0, s1
        x0 = x[:, [1], :]               #p
        x1 = x[:, [2], :]               #c
        x2 = x[:, [0, 1, 2, 3], :]      #m, p, c, s0
        x3 = x[:, [3, 4], :]            #s0, s1
        r0 = x[:, [5], :]               #r0
        r1 = x[:, [6], :]               #r1
        r2 = x[:, [7], :]               #r2
        #extract features
        x0 = self.fe_p(x0)
        x1 = self.fe_c(x1)
        x2 = self.fe_mpcs0(x2)
        x3 = self.fe_s01(x3)
        r0 = self.fe_r0(r0)
        r1 = self.fe_r1(r1)
        r2 = self.fe_r2(r2)
        #connect internal nodes
        i0 = torch.cat((x0, x1, x2, r0, r1), 1)
        x2 = self.ic0(i0)
        i1 = torch.cat((x2, x3, r2), 1)
        x3 = self.ic1(i1)
        #merge information
        m015 = torch.cat((x0, x2, x3, r0), 1)
        m2 = torch.cat((x1, x2, x3, r1), 1)
        m3 = torch.cat((x2, x3), 1)
        m4 = x3
        m0 = self.m0(m015)
        m1 = self.m1(m015)
        m2 = self.m2(m2)
        m3 = self.m3(m3)
        m4 = self.m4(m4)
        m5 = self.m5(m015)
        m0 = m0.view(-1, 400)
        m1 = m1.view(-1, 400)
        m2 = m2.view(-1, 400)
        m3 = m3.view(-1, 400)
        m4 = m4.view(-1, 400)
        m5 = m5.view(-1, 400)
        #predict fault
        y0 = self.fp0(m0)
        y1 = self.fp1(m1)
        y2 = self.fp2(m2)
        y3 = self.fp3(m3)
        y4 = self.fp4(m4)
        y5 = self.fp5(m5)
        y  = torch.cat((y0, y1, y2, y3, y4, y5), 1)
        return y
