"""
influence graph based structrual convolutional neural network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class dscnn_diagnoser(nn.Module):
    """
    influence graph based structrual convolutional neural network diagnoser
    """
    def __init__(self):
        super(dscnn_diagnoser, self).__init__()
        window = 5
        #feature extractor
        self.fe0         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe1         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe2         = nn.Sequential(
                            nn.Conv1d(4, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe3         = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )
        #internal connector
        self.ic0         = nn.Sequential(
                            nn.Conv1d(60, 20, 1),
                            nn.ReLU(),
                          )

        self.ic1         = nn.Sequential(
                            nn.Conv1d(40, 20, 1),
                            nn.ReLU(),
                          )
        #information merger
        self.m0          = nn.Sequential(
                            nn.Conv1d(3*20, 20, 1),
                            nn.ReLU(),
                          )

        self.m1          = nn.Sequential(
                            nn.Conv1d(3*20, 20, 1),
                            nn.ReLU(),
                          )

        self.m2          = nn.Sequential(
                            nn.Conv1d(3*20, 20, 1),
                            nn.ReLU(),
                          )
        
        self.m3          = nn.Sequential(
                            nn.Conv1d(2*20, 20, 1),
                            nn.ReLU(),
                          )
        
        self.m4          = nn.Sequential(
                            nn.Conv1d(20, 20, 1),
                            nn.ReLU(),
                          )
        
        self.m5          = nn.Sequential(
                            nn.Conv1d(3*20, 20, 1),
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
        x0 = x[:, [1], :]               #p
        x1 = x[:, [2], :]               #c
        x2 = x[:, [0, 1, 2, 3], :]      #m
        x3 = x[:, [3, 4], :]            #a
        #extract features
        x0 = self.fe0(x0)
        x1 = self.fe1(x1)
        x2 = self.fe2(x2)
        x3 = self.fe3(x3)
        #connect internal nodes
        i0 = torch.cat((x0, x1, x2), 1)
        x2 = self.ic0(i0)
        i1 = torch.cat((x2, x3), 1)
        x3 = self.ic1(i1)
        #merge information
        m015 = torch.cat((x0, x2, x3), 1)
        m2 = torch.cat((x1, x2, x3), 1)
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
