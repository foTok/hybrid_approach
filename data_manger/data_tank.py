"""
data tank to manage data
"""

from collections import defaultdict
import torch
import time
from torch.autograd import Variable
import numpy as np
from utilities import read_data


class DataTank():
    """
    store and manage data
    """
    def __init__(self):
        self.fault_type = []
        self.data = defaultdict(list)

    def set_fault_type(self, fault_type):
        """
        set the fault type, a string vector
        """
        self.fault_type = fault_type

    def read_data(self, file_name, **kwargs):
        """
        read data and store them in self.normal and self.fault
        """
        fault_type = None if "fault_type" not in kwargs else kwargs["fault_type"]
        step_len = None if "step_len" not in kwargs else kwargs["step_len"]
        split_point = None if "split_point" not in kwargs else kwargs["split_point"]
        normal, fault = read_data(file_name, step_len, split_point)

        fault_vetor = [0]*len(self.fault_type)
        #normal data
        for i in normal:
            self.data[tuple(fault_vetor)].append(i)
        #fault data
        if fault_type is not None:
            fault_type = fault_type if type(fault_type) == list else [fault_type]
            for i in fault_type:
                assert i in self.fault_type
                index = self.fault_type.index(i)
                fault_vetor[index] = 1
            for i in fault:
                self.data[tuple(fault_vetor)].append(i)

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
        warning: bathc here is the number of each fault/normal data
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
                #flatten_data = [val for sublist in chosen_data for val in sublist]
                flatten_data = [sublist[1] for sublist in chosen_data]
                input_data.append(flatten_data)
                target.append(list(mode))
        return torch.Tensor(input_data), torch.Tensor(target)
