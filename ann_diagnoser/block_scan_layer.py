"""
this file define the relation scan ann diagnoser for bpsk system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class MinBlcokScan(nn.Module):
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
        super(MinBlcokScan, self).__init__()
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

    def forward(self, batch_x):
        """
        batch_x: input, [batch, data_size[0]*data_size(1)]
        batch_y: output, [batch, data_size[1] * sum(kernel_size)]
        """
        batch_y = Variable(torch.Tensor(len(batch_x), sum(self.kernel_size)*self.data_size[1]))
        #x: input, size(x) = data_size[0]*data_size(1)
        #y: output, size(y) = data_size[1] * sum(kernel_size)
        for x, batch in zip(batch_x, range(len(batch_x))):
            x = x.view(self.data_size[0], self.data_size[1])
            y = Variable(torch.Tensor(sum(self.kernel_size), self.data_size[1]))
            padding1 = int((self.window_size - 1) / 2)
            parameter_index = -1
            for relation, size in zip(self.dim_relation, self.kernel_size):
                indices = Variable(torch.LongTensor(relation))
                data= torch.index_select(x, 0, indices)
                for _ in range(size):
                    parameter_index = parameter_index + 1
                    kernel = self.parameter_list[parameter_index]
                    for k in range(self.data_size[1]):
                        start = k-padding1
                        end = k+self.window_size-padding1
                        if start >=0 and end <= self.data_size[1]:
                            select_data = data[:, start:end]
                            select_kernel = kernel
                        elif start < 0:
                            select_data = data[:, :end]
                            select_kernel = kernel[:, -start:]
                        elif end > self.data_size[1]:
                            select_data = data[:, start:]
                            select_kernel = kernel[:, :self.data_size[1]-start]
                        y[parameter_index, k] = torch.sum(select_data*select_kernel)
            y = y.view(-1, 1)
            batch_y[batch] = y
        return batch_y


class FullScan(nn.Module):
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
        super(FullScan, self).__init__()
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