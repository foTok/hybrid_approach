"""
hybrid influence graph based structrual convolutional neural network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class higsecnn_diagnoser(nn.Module):
    """
    hybrid influence graph based structrual convolutional neural network diagnoser
    """
    def __init__(self):
        super(higsecnn_diagnoser, self).__init__()
        window = 5
        #embedded part
        self.fe_r0        = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )
        self.higsecnn0    = higsecnn_sub0(self.fe_r0)
        self.higsecnn1    = higsecnn_sub1(self.fe_r0)

    def diagnoser0(self, x):
        x   = self.higsecnn0(x)
        return x

    def diagnoser1(self, x):
        x   = self.higsecnn1(x)
        return x

    def parameters0(self):
        para = self.higsecnn0.parameters()
        return para

    def parameters1(self):
        para = self.higsecnn1.parameters()
        return para

    def forward(self, x):
        x   = self.higsecnn0(x)
        return x


#sub model zero
class higsecnn_sub0(nn.Module):
    """
    use the sub model to predict 6 faults
    """
    def __init__(self, embeded):
        super(higsecnn_sub0, self).__init__()
        self.fe_r0         = embeded
        window = 5
        #feature extractor
        self.fe_p         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe_c         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe_r1        = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe_pcmr01    = nn.Sequential(
                            nn.Conv1d(5, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe_ma        = nn.Sequential(
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe_r2        = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        #internal connector
        self.ic0         = nn.Sequential(
                            nn.Conv1d(20*5, 20, 1),
                            nn.ReLU(),
                          )

        self.ic1         = nn.Sequential(
                            nn.Conv1d(20*3, 20, 1),
                            nn.ReLU(),
                          )
        #information merger
        self.m0          = nn.Sequential(
                            nn.Conv1d(4*20, 20, 1),
                            nn.ReLU(),
                          )

        self.m1          = nn.Sequential(
                            nn.Conv1d(4*20, 20, 1),
                            nn.ReLU(),
                          )

        self.m2          = nn.Sequential(
                            nn.Conv1d(4*20, 20, 1),
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
                            nn.Conv1d(4*20, 20, 1),
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
        x2 = x[:, [1, 2, 3, 5, 6], :]   #p, c, m, r0, r1
        x3 = x[:, [3, 4], :]            #m, a
        r0 = x[:, [5], :]               #r0
        r1 = x[:, [6], :]               #r1
        r2 = x[:, [7], :]               #r2
        #extract features
        x0 = self.fe_p(x0)
        x1 = self.fe_c(x1)
        x2 = self.fe_pcmr01(x2)
        x3 = self.fe_ma(x3)
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


#sub model one
class higsecnn_sub1(nn.Module):
    """
    use the sub model to predict 6 faults
    """
    def __init__(self, embeded):
        super(higsecnn_sub1, self).__init__()
        self.fe_r0      = embeded
        #predictor
        self.fp         = nn.Sequential(
                          nn.Linear(20*20, 40),
                          nn.ReLU(),
                          nn.BatchNorm1d(40),
                          nn.Linear(40, 3),
                          nn.Sigmoid(),
                        )

    def forward(self, x):
        #x: batch × 1 × 100
        x = self.fe_r0(x)
        y = self.fp(x)
        return y
