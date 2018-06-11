"""
unit test for graph model
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import unittest
import numpy as np
from graph_model.utilities import check_loop
from graph_model.utilities import und2od

class TestGraphModel(unittest.TestCase):
    """
    test graph_model/utilities
    """
    def test_check_loop(self):
        cb = []
        check_loop(cb, (0,1))
        self.assertEqual(cb, [[0, 1]])
        check_loop(cb, (1,2))
        self.assertEqual(cb, [[0, 1, 2]])
        check_loop(cb,(3,4))
        self.assertEqual(cb, [[0, 1, 2], [3, 4]])
        check_loop(cb, (2,3))
        self.assertEqual(cb, [[0, 1, 2, 3, 4]])

    def test_und2od(self):
        edges = [(0,1), (2,1), (1,3), (4,2)]
        order = [0, 1, 2, 3, 4]
        graph = und2od(edges, order)
        graph0 = np.zeros((5,5))
        graph0[0,1] = 1
        graph0[1,2] = 1
        graph0[1,3] = 1
        graph0[2,4] = 1
        for i in range(5):
            for j in range(5):
                self.assertEqual(graph[i,j], graph0[i,j])

if __name__ == '__main__':
    unittest.main(exit=False)
