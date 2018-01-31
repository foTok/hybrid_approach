"""
this file define the relation scan ann diagnoser for bpsk system
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



class DiagnoerMinBlcokScan(nn.Module):
    """
    The basic diagnoser for Minimal Block Scan
    """

    def __init__(self, data_len, step_len, b0_num, b1_num, b2_num, b3_num):


    def forward(self, x):
        pass