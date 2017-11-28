"""
bpsk navigation module
"""
import fractions
from math import sqrt
from math import cos
from math import pi
import numpy as np
from .pseudo_generator import Pseudo
from .msg_generator import Msg


class Bpsk:
    """
    r(t) = AD(t)C(t)cos(omiga*t+phi)
    """
    def __init__(self, amplify=None, initial_code=None, code_rate=None, sample_rate=None):
        #normal parameters
        self.amplify = 10 if amplify is None else amplify
        self.code_rate = 1023000 if code_rate is None else code_rate
        self.phi = 0.5 * pi
        self.sample_rate = self.code_rate*10 if sample_rate is None else sample_rate
        self.pseudo = Pseudo(initial_code=initial_code,\
                             code_rate=self.code_rate, sample_rate=self.sample_rate)
        self.msg = Msg(code_rate=self.code_rate, sample_rate=self.sample_rate)
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
            self.pseudo.insert_fault("tma", parameter)
        elif fault == "pseudo_rate":
            self.delta_pseudo_rate = parameter
            self.pseudo.insert_fault("code_rate", parameter)
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
        self.pseudo.set_fault_time(fault_time)

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

    def modulate(self, msg, time):
        """
        modulate one signal point
        """
        code = self.pseudo.sample(time)
        sig = 0
        if time < self.fault_time:
            sig = self.amplify\
                * msg * code\
                * cos(2 * pi * self.code_rate * time + self.phi)
        else:
            sig = (self.amplify + self.delta_amplify)\
                * (sqrt(1-self.alpha) * msg * code + sqrt(self.alpha))\
                * cos((2 * pi * (self.code_rate + self.delta_carrier_rate)) * time + self.phi)
        return sig

    def generate_signal(self, end_time):
        """
        generate similation signal
        the unit of end_time is s(second)
        """
        length = int(end_time*self.sample_rate)
        sample_step = fractions.Fraction(1, self.sample_rate)
        data = np.zeros([length, 3])
        #generate input signal randomly
        for i in range(length):
            time = i * sample_step
            msg = self.msg.sample(time)
            sig = self.modulate(msg, time)
            data[i, 0] = time
            data[i, 1] = msg
            data[i, 2] = sig
        return data
        