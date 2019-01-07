import unittest
import KMeans
import numpy as np


class TestKmeans(unittest.TestCase):

    def test_euclidean_distance(self):
        result = KMeans.euclidean_distance(np.array([[2, 2, 2], [1, 1, 1]]), np.array([1, 1, 1]))
        self.assertEqual(np.round(result[0], 8), 1.73205081)
        self.assertEqual(np.round(result[1], 8), 0.)

if __name__ == '__main__':
    unittest.main()
