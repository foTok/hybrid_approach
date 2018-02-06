"""
this file define the relation scan ann diagnoser for bpsk system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class DiagnoerMinBlcokInitScan(nn.Module):
    """
    The basic diagnoser for Minimal Block Scan
    """

    def __init__(self, data_size, window_size, dim_relation, kernel_size):
        """
        tuple data_size: (int number, int length), the size of data
                         data_size[0]: the number of data dimensionals
                         data_size[1]: the length of data in each dimensionals
        int window_size: the size of scan window
        list dim_relation: the relationship among the dimensionals
                           for example: [[1], [2], [0, 1, 2, 3], [3, 4]]
        list kernel_size: the size of kernel
                          for example: [3, 3, 3, 3]
        """
        super(DiagnoerMinBlcokInitScan, self).__init__()
        assert len(dim_relation) == len(kernel_size)
        self.data_size = data_size
        self.window_size = window_size
        self.dim_relation = dim_relation
        self.kernel_size = kernel_size
        self.parameter_list = []
        for relation, size in zip(dim_relation, kernel_size):
            for _ in range(size):
                tmp_para = Parameter(torch.rand(len(relation), window_size))
                self.parameter_list.append(tmp_para)

    def forward(self, x):
        """
        x: input, size(x) = data_size[0]*data_size(1)
        y: output, size(y) = data_size[1] * sum(kernel_size)
        """
        x = x.view(self.data_size[0], self.data_size(1))
        y = torch.Tensor(sum(self.kernel_size), self.data_size[1])
        padding1 = int((self.window_size - 1) / 2)
        padding2 = self.window_size - 1 - padding1
        for relation, size, i in zip(self.dim_relation, self.kernel_size, range(len(self.kernel_size))):
            for j in range(size):
                for k in range(len(self.data_size[1])):
                    if k < self.window_size - padding1:
                        pass
                    elif k > self.data_size[1] - padding2:
                        pass
                    else:
                        pass
        y = y.view(-1, 1)
        return y
