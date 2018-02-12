"""
pooling block scan results
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import ceil

class BlockMaxPooling(nn.Module):
    """
    pooling block scan
    """
    def __init__(self, data_len, step_len, pooling_size):
        """
        int data_len: the length of data
        int step_len: the length of step
        int pooling_size: the length of pooling
        """
        super(BlockMaxPooling, self).__init__()
        self.data_len = data_len
        self.step_len = step_len
        self.pooling_size = pooling_size
        self.feature_num = int(self.data_len / self.step_len)
        self.pooling_len = ceil(step_len / pooling_size)

    def forward(self, batch_x):
        """
        batch_x: [batch, self.feature_num * self.step_len]
        batch_y: [batch, self.feature_num * self.pooling_len]
        """
        batch_y = Variable(torch.Tensor(len(batch_x), self.feature_num * self.pooling_len))
        for x, batch in zip(batch_x, range(len(batch_x))):
            x = x.view(self.feature_num, self.step_len)
            y = Variable(torch.FloatTensor(self.feature_num, self.pooling_len))
            for i in range(self.feature_num):
                for j in range(self.pooling_len):
                    start = j*self.pooling_size
                    end = (j+1)*self.pooling_size
                    y[i, j] = torch.max(x[i, start:end])
            y = y.view(-1, 1)
            batch_y[batch] = y
        return batch_y
