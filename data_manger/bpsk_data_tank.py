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
        #map from mode to data
        self.map = defaultdict(list)

    def read_data(self, file_name, **kwargs):
        """
        read data and store them
        """
        step_len = None if "step_len" not in kwargs else kwargs["step_len"]
        split_point = None if "split_point" not in kwargs else kwargs["split_point"]
        snr = None if "snr" not in kwargs else kwargs["snr"]
        normal, fault = read_data(file_name, step_len, split_point, snr)
        list_fault, list_parameters = parse_filename(file_name)

        mode = [0, 0, 0, 0, 0, 0]
        #para = [0, 0, 0, 0, 0, [0, 0]]
        para = [0, 0, 0, 0, 0, 0, 0]
        #normal data
        for i in normal:
            self.map[tuple(mode)].append(len(self.input))
            self.input.append(i)
            self.mode.append(tuple(mode))
            self.para.append(tuple(para))

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
            self.map[tuple(mode)].append(len(self.input))
            self.input.append(i)
            self.mode.append(tuple(mode))
            self.para.append(tuple(para))

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

    def random_batch_fault(self, batch, single_fault=10, two_fault=10):
        """
        choose some data and return in torch Tensor
        """
        #fault number
        fault_num = {}
        for k in range(6):
            mode_vector = [0, 0, 0, 0, 0, 0]
            mode_vector[k] = 1
            mode_vector = tuple(mode_vector)
            fault_num[mode_vector] = single_fault
        for k in range(6):
            for j in range(k+1, 6):
                mode_vector = [0, 0, 0, 0, 0, 0]
                mode_vector[k] = 1
                mode_vector[j] = 1
                mode_vector = tuple(mode_vector)
                fault_num[mode_vector] = two_fault
        normalization = 0
        for m in fault_num:
            normalization = normalization + fault_num[m]
        for m in fault_num:
            fault_num[m] = fault_num[m] * batch // normalization
        batch = 0
        for m in fault_num:
            batch = batch + fault_num[m]
        #input – input tensor (minibatch x in_channels x iLen)
        #random init
        input_data = Variable(torch.randn(batch, self.feature_num(), self.step_len()))
        mode = Variable(torch.randn(batch, 6))
        para = Variable(torch.randn(batch, 7))
        #counter
        i = -1
        for m in fault_num:
            len_data = len(self.map[m])
            for _ in range(fault_num[m]):
                i = i + 1
                index = int(np.random.random() * len_data)
                index = self.map[m][index]
                signal = self.input[index]
                input_data[i] = torch.from_numpy(signal)
                mode[i] = torch.Tensor(self.mode[index])
                para[i] = torch.Tensor(self.para[index])
        return input_data, mode, para

    def random_batch(self, batch, normal=0, single_fault=10, two_fault=10):
            """
            choose some data and return in torch Tensor
            normal: the proportion for normal
            single_fault: the weight for single fault
            two_fault: the weight for two faults
            """
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
                    fault_num[mode_vector] = 10
            normalization = 0
            for m in fault_num:
                normalization = normalization + fault_num[m]
            for m in fault_num:
                fault_num[m] = int(fault_num[m] * batch * (1 - normal)) // normalization

            all_fault = 0
            for m in fault_num:
                all_fault = all_fault + fault_num[m]
            mode_vector = [0, 0, 0, 0, 0, 0]
            mode_vector = tuple(mode_vector)
            fault_num[mode_vector] = int(batch * normal)
            #input – input tensor (minibatch x in_channels x iLen)
            #random init
            input_data = Variable(torch.randn(batch, self.feature_num(), self.step_len()))
            mode = Variable(torch.randn(batch, 6))
            para = Variable(torch.randn(batch, 7))
            #counter
            i = -1
            for m in fault_num:
                len_data = len(self.map[m])
                for _ in range(fault_num[m]):
                    i = i + 1
                    index = int(np.random.random() * len_data)
                    index = self.map[m][index]
                    signal = self.input[index]
                    input_data[i] = torch.from_numpy(signal)
                    mode[i] = torch.Tensor(self.mode[index])
                    para[i] = torch.Tensor(self.para[index])
            return input_data, mode, para
