import unittest
from GaussianMixtureModel import GaussianMixtureModel
import numpy as np
import os
TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'mickey_mouse.csv')
class TestGaussianMixtureModel(unittest.TestCase):

    def test_init_values(self):
        input_data = np.array([[4, 4],
                               [0, 0],
                               [8, 7],
                               [-2, -2],
                               [-3, -3],
                               [-4, -4],
                               [-6, -9],
                               [-8, -8]
                               ])
        input_data_1 = np.array([[-2.1, 3],
                                 [-1, 1.1],
                                 [4.3, 0.12]])

        input_data_2 = np.array([[-4, -4],
                                [-6, -6],
                                [-8, -8]])

        data = np.genfromtxt(TESTDATA_FILENAME, delimiter=',')
        cluster_number = 3
        gmm = GaussianMixtureModel(cluster_number=cluster_number, data=data)
        print(np.cov(input_data.T))
        print('means: ', gmm.clusters_means)
        print('cluster_number', gmm.cluster_number)
        print('priors: ', gmm.clusters_priors)
        print('covariances: \n')
        for i in range(gmm.cluster_number):
            print(gmm.clusters_covariances[i])
        means = gmm.clusters_means
        cluster_number = gmm.cluster_number
        priors = gmm.clusters_priors
        covariances = gmm.clusters_covariances
        # ----------------------------------------
        cluster_probability_matrix = gmm.cluster_probability_matrix
        new_cluster_probability_m = gmm._compute_probability_matrix(input_data=data,
                                                                    clusters_means=means,
                                                                    clusters_priors=priors,
                                                                    clusters_covariances=covariances,
                                                                    cluster_probability_matrix=cluster_probability_matrix)
        print(new_cluster_probability_m)


        cluster_probability_matrix = np.array([[0.25, 0.5, 0.25],
                                               [0.5, 0.5, 0.],
                                               [0.125, 0.125, 0.750]])
        res1 = gmm._compute_clusters_priors(cluster_probability_matrix)
        print(res1)

        res2 = gmm._compute_clusters_sizes(cluster_probability_matrix)
        print(res2)


        cluster_probability_matrix_2= np.array([[0.5, 0.5],
                                               [0.5, 0.5],
                                               [0.5, 0.5]])
        test_data = np.array([[8, 2],
                              [2, 6],
                              [2, 2]])
        res3 = gmm._compute_clusters_means(cluster_probability_matrix_2, test_data)
        print(res3)
        gmm.fit()


if __name__ == '__main__':
    unittest.main()
