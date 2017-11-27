"""
bpsk navigation module
"""
from pseudo_generator import Pseudo
import numpy as np
import fractions

class Bpsk:
    """
    r(t) = AD(t)C(t)cos(omiga*t+phi)
    """
    def __init__(self, amplify=None, initial_code=None, rate=None, sample_rate=None):
        #normal parameters
        self.amplify = 10 if amplify is None else amplify
        self.omiga = 1023000 if rate is None else rate
        self.phi = 0.0
        self.sample_rate = self.omiga*10 if sample_rate is None else sample_rate
        self.pseudo = Pseudo(initial_code, self.sample_rate)
        #fault parameters
        self.delta_amplify = 0
        self.delta_pseudo_tma = 0
        self.delta_pseudo_rate = 0
        self.delta_carrier_rate = 0
        self.alpha = 0
        self.fault_time = float("Inf")

    def insert_fault(self, fault, parameter):
        """
        insert fault parameters
        """
        if fault == "amplify":
            self.delta_amplify = parameter
        elif fault == "tma":
            self.delta_pseudo_tma = parameter
            self.pseudo.insert_fault("tma",parameter)
        elif fault == "pseudo_rate":
            self.delta_pseudo_rate = parameter
            self.pseudo.insert_fault("code_rate",parameter)
        elif fault == "carrier_rate":
            self.delta_carrier_rate = parameter
        elif fault == "carrier_leak":
            self.alpha = parameter
        else:
            print("Unknown Fault!")

    def set_fault_time(self, fault_time):
        """
        set the fault time
        """
        self.fault_time = fault_time

    def clear_faults(self):
        """
        clear all the faults
        """
        self.delta_amplify = 0
        self.delta_pseudo_tma = 0
        self.delta_pseudo_rate = 0
        self.delta_carrier_rate = 0
        self.alpha = 0
        self.fault_time = float("Inf")
        self.pseudo.clear_faults()

    def re_init(self):
        """
        reset the bpsk signal generator
        """
        self.clear_faults()
        self.pseudo.re_init()

    def generate_signal(self, fault_time, end_time):
        """
        generate similation signal, t is the end time
        """
        sample_step = fractions.Fraction(1,self.sample_rate)
        #TODO
