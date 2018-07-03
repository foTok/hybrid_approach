"""
This file uses CNN method to diagnosis BPSK system with one variable
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SimpleDiagnoer(nn.Module):
    """
    The simple diagnoser constructed by CNN
    """
    def __init__(self):
        super(SimpleDiagnoer, self).__init__()
        window = 5
        self.CNN = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.FC = nn.Sequential(
                            nn.Linear(20*20, 4*64),
                            nn.ReLU(),
                            nn.BatchNorm1d(4*64),
                            nn.Linear(4*64, 6),
                            nn.Sigmoid()
                          )

    def forward(self, x):
        x = self.CNN(x)
        x = x.view(-1, 20*20)
        x = self.FC(x)
        return x
