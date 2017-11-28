"""
a pseudo code generator
"""

import fractions

class Pseudo:
    """
    Generate the pseudo code by shifting
    Faults parameter are !!! Relative Values !!!
    """
    def __init__(self, initial_code=None, code_rate=None, sample_rate=None):
        self.initial_code = [-1, 1, -1, -1, 1, -1, 1] if initial_code is None else initial_code
        self.code_rate = 1023000 if code_rate is None else code_rate
        self.sample_rate = 10230000 if sample_rate is None else sample_rate
        self.code = self.initial_code
        self.initial_time = fractions.Fraction(0)
        #fault parameters
        self.delta_code_rate = 0
        self.delta_tma = 0
        self.fault_time = float("Inf")

    def sample(self, time):
        """
        get a sample point from the pseudo generator
        time must increase from 0 to end, if jump to some point directly,
        the code will not be shifted right and the sample value will be wrong.
        """
        if time < self.fault_time:
            time_step = fractions.Fraction(1, self.code_rate)
        else:
            time_step = fractions.Fraction(1, int(self.code_rate * (1 + self.delta_code_rate)))

        if (time - self.initial_time) > time_step:
            self.initial_time = self.initial_time + time_step
            self.code = self.code[-1:] + self.code[:-1]

        if time < self.fault_time:
            return self.code[-1]

        if self.delta_tma == 0:
            return self.code[-1]
        else:
            if (self.code[-1] == -1)\
           and ((time - self.initial_time) <= (time_step * self.delta_tma)):
                return self.code[0]
            else:
                return self.code[-1]

    def insert_fault(self, fault, parameter):
        """
        insert fault
        """
        if fault == "code_rate":
            self.delta_code_rate = parameter
        elif fault == "tma":
            self.delta_tma = parameter
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
        self.delta_code_rate = 0
        self.delta_tma = 0
        self.fault_time = float("Inf")

    def re_init(self):
        """
        reset the pseudo generator.
        the initial code, code rate and sample rate are reserved.
        """
        self.clear_faults()
        self.code = self.initial_code
        self.initial_time = fractions.Fraction(0)
