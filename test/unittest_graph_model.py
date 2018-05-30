"""
unit test for graph model
"""
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import unittest
from graph_model.utilities import check_loop

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

if __name__ == '__main__':
    unittest.main()
