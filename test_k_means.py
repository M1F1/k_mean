import unittest
import k_means
import numpy as np


class TestKmeans(unittest.TestCase):

    def test_create_substrings_idxs_1(self):
        result = k_means.euclidean_distance(np.array([[2, 2, 2], [1, 1, 1]]), np.array([1, 1, 1]))
        self.assertEqual(np.round(result[0], 8), 1.73205081)
        self.assertEqual(np.round(result[1], 8), 0.)

if __name__ == '__main__':
    unittest.main()
