"""
data tank to manage data
"""

from collections import defaultdict
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
