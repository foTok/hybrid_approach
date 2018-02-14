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
    def __init__(self, input_size):
        super(DiagnoerFullConnect, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 4*input_size)
        self.fc2 = nn.Linear(4*input_size, 2*input_size)
        self.fc3 = nn.Linear(2*input_size, input_size)
        self.fc4 = nn.Linear(input_size, int(input_size / 2))
        self.fc5 = nn.Linear(int(input_size / 2), 6)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x
