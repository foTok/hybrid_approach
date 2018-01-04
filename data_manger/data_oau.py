"""
data manager for OAU
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import defaultdict
import torch
import time
from torch.autograd import Variable
import numpy as np
from utilities import read_data
from OAU.OAU_data import get_data


class DataOAU():
    """
    a class the manage OAU data
    """
    def __init__(self):
        self.data = defaultdict(list)        #[mode: [[data], residual]]
        self.sample_flag = {}                #[mode: [[0 0 1, ...], ...]]
        self.test_len = {}

    def set_test(self, ratio):
        """
        set the data to be reserved
        """
        for mode in self.data:
            self.test_len[mode] = int(len(self.data[mode]) * ratio)

    def read_data(self, file_name):
        """
        read data and store them
        """
        _, norm_oau_data, n_residuals, labels = get_data(file_name)
        for data, res, lab in zip(norm_oau_data, n_residuals, labels):
            mode = [0, 0, 0, 0, 0]
            if lab == 0: #normal
                pass
            elif lab == 1: #off
                mode[0] = 1
            elif lab == 2: #transition
                mode[1] = 1
            elif lab == 3: #both dirty
                mode[2] = 1
                mode[3] = 1
            elif lab == 4: #dirty 1
                mode[2] = 1
            elif lab == 5:
                mode[3] = 1
            else:
                mode[4] = 1
            mode = tuple(mode)
            self.data[mode].append([data, res])
        for m in self.data:
            self.test_len[m] = 0
            self.sample_flag[m] = [0] * len(self.data[m])

    def data_size(self):
        """
        the size of mode
        """
        for i in self.data:
            in_size = len(self.data[i][0][0])
            out_size = len(i)
            break
        return in_size, out_size

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
                rand = int(np.random.random() * (len_data - self.test_len[mode]))
                self.sample_flag[mode][rand] = 1                #set sampled flag
                chosen_data = the_data[rand]
                input_data.append(chosen_data[0])
                target.append(list(mode))
        return torch.Tensor(input_data), torch.Tensor(target)

    def clear_sample_flag(self):
        """
        clear all the sample flags
        """
        for m in self.data:
            self.sample_flag[m] = [0] * len(self.data[m])

    def unsampled_data(self):
        """
        get all unsampled data
        """
        input_data = []
        target = []
        #type number of unsampled data
        type_num = 0
        for mode in self.sample_flag:
            if self.sample_flag[mode].count(0) != 0:
                print("left mode: " + str(mode))
                type_num = type_num + 1
        print("There are %d types samples left." % type_num)
        for mode in self.data:
            the_data = self.data[mode]
            the_flag = self.sample_flag[mode]
            for data, flag in zip(the_data, the_flag):
                if flag == 0:
                    input_data.append(data[0])
                    target.append(list(mode))
        return torch.Tensor(input_data), torch.Tensor(target)

if __name__ == '__main__':
    file_name = '../OAU/OAU1201.csv'
    data_mana = DataOAU()
    data_mana.read_data(file_name)
    data = data_mana.random_batch(10)
