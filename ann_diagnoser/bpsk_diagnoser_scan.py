"""
this file define the relation scan ann diagnoser for bpsk system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class DiagnoerMinBlcokScan(nn.Module):
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
        super(DiagnoerMinBlcokScan, self).__init__()
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
        parameter_index = -1
        for relation, size in zip(self.dim_relation, self.kernel_size):
            for _ in range(size):
                parameter_index = parameter_index + 1
                for k in range(len(self.data_size[1])):
                    if k < padding1:
                        data = x[relation,:k+self.window_size-padding1]
                        kernel = self.parameter_list[parameter_index][:,:k+self.window_size-padding1]
                    elif k > self.data_size[1] - self.window_size + padding1:
                        data = x[relation,k-padding1:]
                        kernel = self.parameter_list[parameter_index][:,self.window_size+k-self.data_size[1]-padding1:]
                    else:
                        data = x[relation,k-padding1:k+self.window_size-padding1]
                        kernel = self.parameter_list[parameter_index]
                    y[parameter_index, k] = torch.sum(data*kernel)
        y = y.view(-1, 1)
        return y


class DiagnoerScan(nn.Module):
    """
    The basic diagnoser for Minimal Block Scan
    """

    def __init__(self, data_size, window_size, kernel_size):
        """
        tuple data_size: (int number, int length), the size of data
                         data_size[0]: the number of data dimensionals
                         data_size[1]: the length of data in each dimensionals
        int window_size: the size of scan window
        int kernel_size: the size of kernel
        """
        super(DiagnoerScan, self).__init__()
        self.data_size = data_size
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.parameter_list = []
        for _ in range(kernel_size):
            tmp_para = Parameter(torch.rand(data_size[0], window_size))
            self.parameter_list.append(tmp_para)

    def forward(self, x):
        """
        x: input, size(x) = data_size[0]*data_size(1)
        y: output, size(y) = data_size[1] * sum(kernel_size)
        """
        x = x.view(self.data_size[0], self.data_size(1))
        y = torch.Tensor(sum(self.kernel_size), self.data_size[1])
        padding1 = int((self.window_size - 1) / 2)
        for i in range(self.kernel_size):
            for k in range(len(self.data_size[1])):
                if k < padding1:
                    data = x[:,:k+self.window_size-padding1]
                    kernel = self.parameter_list[i][:,:k+self.window_size-padding1]
                elif k > self.data_size[1] - self.window_size + padding1:
                    data = x[:,k-padding1:]
                    kernel = self.parameter_list[i][:,self.window_size+k-self.data_size[1]-padding1:]
                else:
                    data = x[:,k-padding1:k+self.window_size-padding1]
                    kernel = self.parameter_list[i]
                y[i, k] = torch.sum(data*kernel)
        y = y.view(-1, 1)
        return y