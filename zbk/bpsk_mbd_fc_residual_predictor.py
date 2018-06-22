"""
This file uses full connection ann to predict residuals
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class res_predictor(nn.Module):
    """
    predict residuals 1
    """
    def __init__(self, f):
        super(res_predictor, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(f, 6*f),
            nn.ReLU(),
            nn.Linear(6*f, 4*f),
            nn.ReLU(),
            nn.Linear(4*f, 1),
        )

    def forward(self, x):
        x = self.sequence(x)
        return x
