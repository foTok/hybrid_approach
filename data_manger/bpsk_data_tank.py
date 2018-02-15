"""
data tank to manage data
"""

import os
from collections import defaultdict

import torch
import numpy as np
from torch.autograd import Variable
from utilities import read_data


def parse_filename(filename):
    """
    obtain parameters from the filename
    including fault types and the value of parameters
    """
    filename = os.path.split(filename)
    str_s = filename[-1].split("@")
    list_faults = str_s[0]
    list_faults = list_faults.split(",")

    list_parameters = str_s[1].rstrip(".npy")
    list_parameters = list_parameters.split(",")

    return list_faults, list_parameters



class BpskDataTank():
    """
    store and manage data
    """
    def __init__(self):
        self.fault_type = ["tma", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify", "tmb"]
        #input:[batch × feature × step_len]
        #list of numpy ndarray
        self.input = []
        #list of list
        self.mode = []
        #list of list
        self.para = []

    def read_data(self, file_name, **kwargs):
        """
        read data and store them
        """
        step_len = None if "step_len" not in kwargs else kwargs["step_len"]
        split_point = None if "split_point" not in kwargs else kwargs["split_point"]
        normal, fault = read_data(file_name, step_len, split_point)
        list_fault, list_parameters = parse_filename(file_name)

        mode = [0, 0, 0, 0, 0, 0]
        #para = [0, 0, 0, 0, 0, [0, 0]]
        para = [0, 0, 0, 0, 0, 0, 0]
        #normal data
        for i in normal:
            self.input.append(i)
            self.mode.append(mode)
            self.para.append(para)
        #fault data
        #find faults and parameters
        for i, j in zip(list_fault, list_parameters):
            assert i in self.fault_type
            index = self.fault_type.index(i)
            mode[index] = 1
            if j.find("(") != -1:
                j = j.lstrip("(")
                para[5] = float(j)
            elif j.find(")") != -1:
                j = j.rstrip(")")
                para[6] = float(j)
            else:
                para[index] = float(j)
        for i in fault:
            self.input.append(i)
            self.mode.append(mode)
            self.para.append(para)

    def step_len(self):
        """
        return the step_len
        """
        return len(self.input[0][0])

    def feature_num(self):
        """
        return the feature number
        """
        return len(self.input[0])

    def info(self):
        """
        print basic infomation
        """
        print("There are {} data".format(len(self.input)))

    def random_batch(self, batch):
        """
        choose some data and return in torch Tensor
        batch here is the number of each fault/normal data
        Not the whole chosen data
        """
        #input – input tensor (minibatch x in_channels x iH x iW)
        #random init
        input_data = Variable(torch.randn(batch, self.feature_num(), self.step_len()))
        mode = Variable(torch.randn(batch, 6))
        para = Variable(torch.randn(batch, 7))
        #refuse sample
        #fault number
        fault_num = {}
        for k in range(6):
            mode_vector = [0, 0, 0, 0, 0, 0]
            mode_vector[k] = 1
            mode_vector = tuple(mode_vector)
            fault_num[mode_vector] = 10
        for k in range(6):
            for j in range(k+1, 6):
                mode_vector = [0, 0, 0, 0, 0, 0]
                mode_vector[k] = 1
                mode_vector[j] = 1
                mode_vector = tuple(mode_vector)
                fault_num[mode_vector] = 1
        normalization = 0
        for m in fault_num:
            normalization = normalization + fault_num[m]
        for m in fault_num:
            fault_num[m] = fault_num[m] * (batch // 2) // normalization
        all_fault_number = 0
        for m in fault_num:
            all_fault_number = all_fault_number + fault_num[m]
        normal_number = batch - all_fault_number
        fault_num[tuple([0, 0, 0, 0, 0, 0])] = normal_number
        #counter
        i = 0
        while i < batch:
            len_data = len(self.input)
            index = int(np.random.random() * len_data)
            current_mode = tuple(self.mode[index])
            if fault_num[current_mode] > 0:
                i = i + 1
                fault_num[current_mode] = fault_num[current_mode] - 1
                input_data[i] = torch.from_numpy(self.input[index])
                mode[i] = torch.Tensor(self.mode[index])
                para[i] = torch.Tensor(self.para[index])
            else:#refuse
                pass
        return input_data, mode, para
