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
        self.fc1 = nn.Linear(step_len, 4*step_len)
        self.fc2 = nn.Linear(4*step_len, 2*step_len)
        self.fc3 = nn.Linear(2*step_len, step_len)
        self.fc4 = nn.Linear(step_len, int(step_len / 2))
        self.fc5 = nn.Linear(int(step_len / 2), 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x
