"""
data tank to manage data
"""

import os
from collections import defaultdict

import torch
import numpy as np
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
        self.fault_type = ["tma", "tmb", "pseudo_rate", "carrier_rate", "carrier_leak", "amplify"]
        self.data = defaultdict(list)
        self.para = defaultdict(list)

    def read_data(self, file_name, **kwargs):
        """
        read data and store them
        """
        step_len = None if "step_len" not in kwargs else kwargs["step_len"]
        split_point = None if "split_point" not in kwargs else kwargs["split_point"]
        normal, fault = read_data(file_name, step_len, split_point)
        list_fault, list_parameters = parse_filename(file_name)

        fault_vetor = [0, 0, 0, 0, 0, 0]
        para_vector = [0, [0, 0], 0, 0, 0, 0]
        #normal data
        for i in normal:
            self.data[tuple(fault_vetor)].append(i)
            self.para[tuple(fault_vetor)].append(para_vector)
        #fault data
        for i, j in zip(list_fault, list_parameters):
            assert i in self.fault_type
            index = self.fault_type.index(i)
            fault_vetor[index] = 1
            if j.find("(") != -1:
                j = j.lstrip("(")
                para_vector[1][0] = float(j)
            elif j.find(")") != -1:
                j = j.rstrip(")")
                para_vector[1][1] = float(j)
            else:
                para_vector[index] = float(j)
        for i in fault:
            self.data[tuple(fault_vetor)].append(i)
            self.para[tuple(fault_vetor)].append(para_vector)

    def choose_data_randomly(self, fault_type, num):
        """
        choose num data in the specific (fault_type) fault randomly
        """
        fault_type = tuple(fault_type) if type(fault_type) == list else fault_type
        chosen_data = []
        if fault_type in self.data:
            the_data = self.data[fault_type]
            len_data = len(the_data)
            for _ in range(num):
                rand = int(np.random.random() * len_data)
                chosen_data.append(the_data[rand])
        return chosen_data

    def step_len(self):
        """
        return the step_len
        """
        normal_state = tuple([0] * len(self.fault_type))
        normal_data = self.data[normal_state]
        return len(normal_data[0])

    def random_batch(self, batch):
        """
        choose some data and return in torch Tensor
        warning: batch here is the number of each fault/normal data
        Not the whole chosen data
        """
        #np.random.seed(int(time.time()))
        input_data = []
        target = []
        for mode in self.data:
            the_data = self.data[mode]
            len_data = len(the_data)
            for _ in range(batch):
                rand = int(np.random.random() * len_data)
                chosen_data = the_data[rand]
                flatten_data = [i for sublist in chosen_data for i in sublist]
                input_data.append(flatten_data)
                target.append(list(mode))
        return torch.Tensor(input_data), torch.Tensor(target)

    def random_normal_fault_batch(self, batch):
        """
        choose some data and return in torch Tensor
        warning: batch here is the number of each fault/normal data
        Not the whole chosen data
        """
        #np.random.seed(int(time.time()))
        input_data = []
        target = []
        for mode in self.data:
            the_data = self.data[mode]
            len_data = len(the_data)
            if sum(mode) == 0:
                num = batch
            else:
                num = int(batch/(len(self.data)-1))
            for _ in range(num):
                rand = int(np.random.random() * len_data)
                chosen_data = the_data[rand]
                flatten_data = [i for sublist in chosen_data for i in sublist]
                input_data.append(flatten_data)
                if sum(mode) == 0:
                    target.append(0)
                else:
                    target.append(1)
        return torch.Tensor(input_data), torch.Tensor(target)
