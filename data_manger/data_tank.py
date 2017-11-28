"""
data tank to manage data
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from collections import defaultdict
from utilities import read_data


class DataTank():
    """
    store and manage data
    """
    def __init__(self):
        self.normal = []                    #[data1, data2...]
        self.fault_type = []
        self.fault = defaultdict(list)

    def set_fault_type(self, fault_type):
        """
        set the fault type, a string vector
        """
        self.fault_type = fault_type

    def read_data(self, file_name, **kwargs):
        """
        read data and store them in self.normal and self.fault
        """
        fault_type = None if "fault_type" not in kwargs else kwargs["fault"]
        step_len = None if "step_len" not in kwargs else kwargs["step_len"]
        split_point = None if "split_point" not in kwargs else kwargs["split_point"]
        normal, fault = read_data(file_name, step_len, split_point)
        #normal data
        for i in normal:
            self.normal.append(i)
        #fault data
        fault_vetor = [0]*len(self.fault_type)
        for i in fault_type:
            assert i in self.fault_type
            index = self.fault_type.index(i)
            fault_vetor[index] = 1
        for i in fault:
            self.fault[fault_vetor].append(i)
