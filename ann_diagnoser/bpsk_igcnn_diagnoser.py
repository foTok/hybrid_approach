"""
influence graph based CNN diagnoser
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class dcnn_diagnoser(nn.Module):
    """
    using influence graph to extract features from historical data and predict fault directly after merging.
    """
    def __init__(self):
        super(dcnn_diagnoser, self).__init__()
        window = 5
        #influence graph: dim_relation = [[1], [2], [1, 2, 3], [3, 4]]
        self.p_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.c_sequence = nn.Sequential(
                            nn.Conv1d(1, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.m_sequence = nn.Sequential(
                            nn.Conv1d(4, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.a_sequence = nn.Sequential(
                            nn.Conv1d(3, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.merge_sequence = nn.Sequential(
                            nn.Conv1d(4*20, 40, 1),
                            nn.ReLU(),
                        )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(40*20, 80),
                            nn.ReLU(),
                            nn.BatchNorm1d(80),
                            nn.Linear(80, 6),
                            nn.Sigmoid(),
                          )

    def forward(self, x):
        x0 = x[:, [1], :]
        x1 = x[:, [2], :]
        x2 = x[:, [1, 2, 3], :]
        x3 = x[:, [3, 4], :]
        x0 = self.p_sequence(x0)
        x1 = self.c_sequence(x1)
        x2 = self.m_sequence(x2)
        x3 = self.a_sequence(x3)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = x.view(-1, 4*20, 20)
        x = self.merge_sequence(x)
        x = x.view(-1, 40*20)
        x = self.fc_sequence(x)
        return x
