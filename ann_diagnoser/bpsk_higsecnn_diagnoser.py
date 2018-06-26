"""
hybrid influence graph based structrual convolutional neural network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class higsecnn_diagnoser(nn.Module):
    """
    hybrid influence graph based structrual convolutional neural network diagnoser
    """
    def __init__(self):
        super(higsecnn_diagnoser, self).__init__()
        window = 5
        self.fe_r0        = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                  )
        self.diagnoser0   = sub_diagnoser0(self.fe_r0)
        self.diagnoser1   = sub_diagnoser1(self.fe_r0)
        self.merger0      = nn.Linear(2, 1, bias=False)
        self.merger1      = nn.Linear(2, 1, bias=False)
        self.merger5      = nn.Linear(2, 1, bias=False)

    def parameters0(self):
        return self.diagnoser0.parameters()

    def parameters1(self):
        return self.diagnoser1.parameters()

    def freeze_sub0(self):
        for para in self.diagnoser0.parameters():
            para.requires_grad=False

    def freeze_sub1(self):
        for para in self.diagnoser1.parameters():
            para.requires_grad=False

    def forward0(self, x):
        x  = self.diagnoser0(x)
        x   = F.sigmoid(x)
        return x

    def forward1(self, x):
        x = self.diagnoser1(x)
        x = F.sigmoid(x)
        return x

    def merge_forward0(self, x):
        x0  = self.diagnoser0(x)
        x00 = x0[:,[0]]
        x10 = x0[:,[1]]
        x2  = x0[:,[2]]
        x3  = x0[:,[3]]
        x4  = x0[:,[4]]
        x50 = x0[:,[5]]
        x1  = self.diagnoser1(x[:, [5], :])
        x01 = x1[:, [0]]
        x11 = x1[:, [1]]
        x51 = x1[:, [2]]
        x0  = self.merger0(torch.cat((x00, x01), 1))
        x1  = self.merger1(torch.cat((x10, x11), 1))
        x5  = self.merger5(torch.cat((x50, x51), 1))
        x   = torch.cat((x0, x1, x2, x3, x4, x5), 1)
        x   = F.sigmoid(x)
        return x

    def merge_forward1(self, x):
        x   = self.diagnoser1(x)
        x01 = x[:, [0]]
        x11 = x[:, [1]]
        x51 = x[:, [2]]
        shape = x01.shape
        x00 = torch.zeros(shape)
        x10 = torch.zeros(shape)
        x50 = torch.zeros(shape)
        x0  = self.merger0(torch.cat((x00, x01), 1))
        x1  = self.merger1(torch.cat((x10, x11), 1))
        x5  = self.merger5(torch.cat((x50, x51), 1))
        x   = torch.cat((x0, x1, x5), 1)
        x   = F.sigmoid(x)
        return x

    def forward(self, x):
        x0  = self.diagnoser0(x)
        x00 = x0[:,[0]]
        x10 = x0[:,[1]]
        x2  = x0[:,[2]]
        x3  = x0[:,[3]]
        x4  = x0[:,[4]]
        x50 = x0[:,[5]]
        x1  = self.diagnoser1(x[:, [5], :])
        x01 = x1[:, [0]]
        x11 = x1[:, [1]]
        x51 = x1[:, [2]]
        x0  = self.merger0(torch.cat((x00, x01), 1))
        x1  = self.merger1(torch.cat((x10, x11), 1))
        x5  = self.merger5(torch.cat((x50, x51), 1))
        x   = torch.cat((x0, x1, x2, x3, x4, x5), 1)
        x   = F.sigmoid(x)
        return x

class sub_diagnoser0(nn.Module):
    def __init__(self, r0):
        super(sub_diagnoser0, self).__init__()
        window = 5
        #feature extractor
        self.fe_p         = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fe_r0        = r0

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
                            nn.Conv1d(3*20, 20, 1),
                            nn.ReLU(),
                          )

        self.m1          = nn.Sequential(
                            nn.Conv1d(3*20, 20, 1),
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
                            nn.Conv1d(1*20, 20, 1),
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
                          )
        self.fp1         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                          )

        self.fp2         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                          )

        self.fp3         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                          )

        self.fp4         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
                          )

        self.fp5         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 1),
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
        m015 = torch.cat((x0, x2, x3), 1)
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
        y0  = self.fp0(m0)
        y1  = self.fp1(m1)
        y2  = self.fp2(m2)
        y3  = self.fp3(m3)
        y4  = self.fp4(m4)
        y5  = self.fp5(m5)
        y   = torch.cat((y0, y1, y2, y3, y4, y5), 1)
        return y

class sub_diagnoser1(nn.Module):
    def __init__(self, r0):
        super(sub_diagnoser1, self).__init__()
        self.fe_r0      = r0
        self.fp         = nn.Sequential(
                            nn.Linear(20*20, 40),
                            nn.ReLU(),
                            nn.BatchNorm1d(40),
                            nn.Linear(40, 3),
                          )

    def forward(self, x):
        r0 = x.view(-1, 1, 100)
        r0 = self.fe_r0(r0)
        r0 = r0.view(-1, 400)
        #predict fault
        y  = self.fp(r0)
        return y
