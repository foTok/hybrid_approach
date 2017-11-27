"""
bpsk navigation module
"""

import numpy as np

class Bpsk:
    """
    r(t) = AD(t)C(t)cos(omiga*t+phi)
    """
    def __init__(self, pseudo, amplify, sample=None):
        #normal parameters
        self.pseudo = pseudo
        self.amplify = amplify
        self.omiga = 1.023*(10**6)
        self.phi = 0.0
        self.sample = self.omiga*10 if sample is None else sample
        #fault parameters
        self.delta_pseudo = 0.0 #self.delta_pseudo belongs to [0,1)
        self.delta_rate = 0.0
        self.delta_omiga = 0.0
        self.alpha = 0.0
        self.delta_amplify = 0.0
        self.fault_time = float('Inf')

    def insert_fault(self, fault, parameter):
        """
        insert fault parameters
        """
        if fault == "pseudo":
            self.delta_pseudo = parameter
        elif fault == "rate":
            self.delta_rate = parameter
        elif fault == "omiga":
            self.delta_omiga = parameter
        elif fault == "alpha":
            self.alpha = parameter
        elif fault == "amplify":
            self.delta_amplify = parameter
        else:
            print("Unknown Fault!")

    def set_fault_time(self, fault_time):
        """
        set the fault time
        """
        self.fault_time = fault_time

    def clear_fault(self):
        """
        clear all the faults
        """
        self.delta_pseudo = 0.0 #self.delta_pseudo belongs to [0,1)
        self.delta_rate = 0.0
        self.delta_omiga = 0.0
        self.alpha = 0.0
        self.delta_amplify = 0.0
        self.fault_time = float('Inf')

    def generate_signal(self, t):
        """
        generate similation signal, t is the end time
        """
        step = 1/self.sample
