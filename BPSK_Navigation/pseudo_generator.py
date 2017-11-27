"""
a pseudo code generator
"""

import fractions

class Pseudo:
    """
    Generate the pseudo code by shifting
    """
    def __init__(self, sample_rate, initial_code=None):
        self.initial_code = [0, 1, 0, 0, 1, 0, 1] if initial_code is None else initial_code
        self.code = self.initial_code
        self.code_rate = 1023000
        self.sample_rate = sample_rate
        self.initial_time = fractions.Fraction(0)
        #fault parameters
        self.delta_code_rate = 0
        self.delta_tma = 0

    def sample(self, time):
        """
        get a sample point from the pseudo generator
        time must increase from 0 to end, if jump to some point directly,
        the code will not be shifted right and the sample value will be wrong.
        """
        time_step = fractions.Fraction(1, self.code_rate+self.delta_code_rate)
        if (time - self.initial_time) > time_step:
            self.initial_time = self.initial_time + time_step
            self.code = self.code[-1:] + self.code[:-1]

        if self.delta_tma == 0:
            return self.code[-1]
        else:
            if (self.code[-1] == 0) and ((time - self.initial_time) < (time_step + self.delta_tma)):
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

    def clear_faults(self):
        """
        clear all the faults
        """
        self.delta_code_rate = 0
        self.delta_tma = 0

    def re_init(self):
        """
        reset the pseudo generator.
        the initial code, code rate and sample rate are reserved.
        """
        self.clear_faults()
        self.code = self.initial_code
        self.initial_time = fractions.Fraction(0)
