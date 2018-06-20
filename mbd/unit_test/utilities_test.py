import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,parentdir)
import unittest
import numpy as np
from mbd.utilities import hypothesis_test
from mbd.utilities import is_new

class TestUtilites(unittest.TestCase):
    def test_hypothesis(self):
        """
        RT
        """
        res     = np.array([1, 2, -4])
        var     = np.array([1,1,1])
        alpha   = 0.99
        result  = hypothesis_test(res, var, alpha)
        self.assertEqual([True, True, False], result)

    def test_is_new(self):
        para_set = {}
        para_set["tma"] = [0.3, 0.4, 0.5]
        para_set["tmb"] = [(0.8*10**6, 7.3*10**6), (1*10**6, 7.3*10**6), (1*10**6, 8*10**6)]
        para_set[("tma","tmb")] = [(0.3, (0.8*10**6, 7.3*10**6))]
        grid = [0.1, 1.41421*10**6]
        #fault tma
        para0 = 0.35
        self.assertFalse(is_new("tma", grid[0], para0, para_set["tma"]))
        para0 = 0.7
        self.assertTrue(is_new("tma", grid[0], para0, para_set["tma"]))
        #fault tmb
        para0 = (1.5*10**6, 8.3*10**6)
        self.assertFalse(is_new("tmb", grid[1], para0, para_set["tmb"]))
        para0 = (1.5*10**6, 10.3*10**6)
        self.assertTrue(is_new("tmb", grid[1], para0, para_set["tmb"]))
        #fault tma, tmb
        para0 = (0.35, (1.5*10**6, 8.3*10**6))
        self.assertFalse(is_new(("tma","tmb"), (grid[0], grid[1]), para0, para_set[("tma","tmb")]))
        para0 = (0.5, (1.5*10**6, 8.3*10**6))
        self.assertTrue(is_new(("tma","tmb"), (grid[0], grid[1]), para0, para_set[("tma","tmb")]))

if __name__ == '__main__':
    unittest.main(exit=False)
