"""
defines the ann used as the diagnoer
implemented by pytorch
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class DiagnoerFullConnect(nn.Module):
    """
    The basic diagnoser constructed by full connection
    """
    def __init__(self, step_len):
        super(DiagnoerFullConnect, self).__init__()
        self.fc1 = nn.Linear(2*step_len, 4*step_len)
        self.fc2 = nn.Linear(4*step_len, 6*step_len)
        self.fc3 = nn.Linear(6*step_len, 4*step_len)
        self.fc4 = nn.Linear(4*step_len, 5)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x
