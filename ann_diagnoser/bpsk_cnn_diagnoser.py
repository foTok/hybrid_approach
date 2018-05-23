"""
This file uses CNN to diagnosis BPSK system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiagnoerCNN(nn.Module):
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self):
        super(DiagnoerCNN, self).__init__()
        window = 3
        self.cnn_sequence = nn.Sequential(
                            nn.Conv2d(1, 40, window, padding=window//3),
                            nn.ReLU(),
                            nn.Conv2d(40, 80, window, padding=window//3),
                            nn.ReLU(),
                            nn.MaxPool2d(window),
                        )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(80 * 33, 40),
                            nn.ReLU(),
                            nn.Linear(40, 6),
                            nn.Sigmoid(),
                        )

    def forward(self, x):
        x = self.cnn_sequence(x)
        x = x.view(-1, 80 * 33)
        x = self.fc_sequence(x)
        return x
