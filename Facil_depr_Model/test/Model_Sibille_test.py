# -*- coding:utf-8 -*-

import unittest
from Model.Model_Sibille import *
from Model.parameters import *


class Test_Model(unittest.TestCase):

    def test_time_simulation(self):

        self.assertLess(t_start, t_end)


if __name__ == '__main__':
    unittest.main()