import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,parentdir)
import unittest
import numpy as np
from mbd.utilities import hypothesis_test

class TestUtilites(unittest.TestCase):
    def test_hypothesis_test(self):
        """
        RT
        """
        res     = np.array([1, 2, -4])
        var     = np.array([1,1,1])
        alpha   = 0.99
        result  = hypothesis_test(res, var, alpha)
        self.assertEqual([True, True, False], result)

if __name__ == '__main__':
    unittest.main(exit=False)
