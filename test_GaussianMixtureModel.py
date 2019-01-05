import unittest
from GaussianMixtureModel import GaussianMixtureModel
import numpy as np

def _compute_mean_2(self, cluster_probability_matrix, input_data):
    return np.dot(cluster_probability_matrix.T, input_data) \
           / np.sum(self.cluster_probability_matrix, axis=0)[:, None]

class TestGaussianMixtureModel(unittest.TestCase):

    def test_compute_mean_2(self):
        input_data = np.array([[4, 4],
                               [2, 2],
                               [8, 8]])
        cluster_probability_matrix = np.array([[0.25, 0.5, 0.25],
                                               [0.5, 0.5, 0.],
                                               [0.125, 0.125, 0.750]])
        print(cluster_probability_matrix.T)
        result = GaussianMixtureModel._compute_mean_2(cluster_probability_matrix=cluster_probability_matrix,
                                                      input_data=input_data)
        print(result)
        com = np.array([3.42857143,
                        3.42857143,
                        3.55555556,
                        3.55555556,
                        7.,
                        7.,
                        ]).reshape(3, 2)
        assert (np.array_equal(np.round(result,5), np.round(com,5)))

if __name__ == '__main__':
    unittest.main()
