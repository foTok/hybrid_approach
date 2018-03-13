"""
This file uses block scan method to diagnosis BPSK system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiagnoerBlockScan(nn.Module):
    """
    The basic diagnoser constructed by block scan
    """
    def __init__(self, step_len):
        super(DiagnoerBlockScan, self).__init__()
        window = 5
        #dim_relation = [[1], [2], [0, 1, 2, 3], [3, 4]]
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
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(4*20*20, 4*64),
                            nn.ReLU(),
                            nn.Linear(4*64, 6),
                            nn.Sigmoid()
                          )

    def forward(self, x):
        x1 = x[:, [1], :]
        x2 = x[:, [2], :]
        x3 = x[:, [0, 1, 2, 3], :]
        x4 = x[:, [3, 4], :]
        x1 = self.p_sequence(x1)
        x2 = self.c_sequence(x2)
        x3 = self.m_sequence(x3)
        x4 = self.a_sequence(x4)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = x.view(-1, 4*20*20)
        x = self.fc_sequence(x)
        return x

class DetectorBlockScan(nn.Module):
    """
    The basic detector constructed by block scan
    """
    def __init__(self, step_len):
        super(DetectorBlockScan, self).__init__()
        window = 5
        #dim_relation = [[1], [2], [0, 1, 2, 3], [3, 4]]
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
                            nn.Conv1d(2, 10, window, padding=window//2),
                            nn.ReLU(),
                            nn.Conv1d(10, 20, window, padding=window//2),
                            nn.ReLU(),
                            nn.MaxPool1d(window)
                          )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(4*20*20, 4*64),
                            nn.ReLU(),
                            nn.Linear(4*64, 1),
                            nn.Sigmoid()
                          )

    def forward(self, x):
        x1 = x[:, [1], :]
        x2 = x[:, [2], :]
        x3 = x[:, [0, 1, 2, 3], :]
        x4 = x[:, [3, 4], :]
        x1 = self.p_sequence(x1)
        x2 = self.c_sequence(x2)
        x3 = self.m_sequence(x3)
        x4 = self.a_sequence(x4)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = x.view(-1, 4*20*20)
        x = self.fc_sequence(x)
        return x

class PredictorBlockScan(nn.Module):
    """
    The basic detector constructed by block scan
    """
    def __init__(self, step_len):
        super(PredictorBlockScan, self).__init__()
        window = 5
        #dim_relation = [[1], [2], [0, 1, 2, 3], [3, 4]]
        self.p_sequence = nn.Sequential(
                            nn.Conv1d(1, 20, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.Conv1d(20, 40, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.MaxPool1d(window)
                          )

        self.c_sequence = nn.Sequential(
                            nn.Conv1d(1, 20, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.Conv1d(20, 40, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.MaxPool1d(window)
                          )

        self.m_sequence = nn.Sequential(
                            nn.Conv1d(4, 20, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.Conv1d(20, 40, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.MaxPool1d(window)
                          )

        self.a_sequence = nn.Sequential(
                            nn.Conv1d(2, 20, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.Conv1d(20, 40, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.MaxPool1d(window)
                          )

        self.merge_sequence = nn.Sequential(
                            nn.Conv1d(4*40, 40, window, padding=window//2),
                            nn.LeakyReLU(),
                            nn.Conv1d(40, 20, window, padding=window//2),
                            nn.LeakyReLU()
                          )

        self.fc_sequence = nn.Sequential(
                            nn.Linear(20*20, 100),
                            nn.LeakyReLU(),
                            nn.Linear(100, 7)
                          )

    def forward(self, x):
        x1 = x[:, [1], :]
        x2 = x[:, [2], :]
        x3 = x[:, [0, 1, 2, 3], :]
        x4 = x[:, [3, 4], :]
        x1 = self.p_sequence(x1)
        x2 = self.c_sequence(x2)
        x3 = self.m_sequence(x3)
        x4 = self.a_sequence(x4)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.merge_sequence(x)
        x = x.view(-1, 20*20)
        x = self.fc_sequence(x)
        return x