"""
defines the ann used as the diagnoer for OAU
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
    def __init__(self, input_num, output_num):
        super(DiagnoerFullConnect, self).__init__()
        self.fc1 = nn.Linear(input_num, 4*input_num)
        self.fc2 = nn.Linear(4*input_num, 2*input_num)
        self.fc3 = nn.Linear(2*input_num, input_num)
        self.fc4 = nn.Linear(input_num, int(input_num / 2))
        self.fc5 = nn.Linear(int(input_num / 2), output_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x