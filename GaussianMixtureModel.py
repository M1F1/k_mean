import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
from k_means import KMeans

# set font size of labels on matplotlib plots
plt.rc('font', size=16)

# set style of plots
sns.set_style('white')

# define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)


class GaussianMixtureModel:
    def __init__(self, cluster_number: int, data: np.ndarray):
        k_means = KMeans(data=data, cluster_number=cluster_number)
        k_means.fit()

        self.cluster_number = cluster_number
        self.data = data

        self.clusters_means = np.zeros((cluster_number, data.shape[1]))
        self.clusters_means = k_means.cluster_data_means

        self.clusters_priors = np.zeros((1, cluster_number))
        self.clusters_priors = np.sum(k_means.cluster_assignment_matrix, axis=0) / data.shape[0]

        self.clusters_covariances = np.zeros((cluster_number, data.shape[1], data.shape[1]))
        for k in range(self.cluster_number):
            idx = np.nonzero(k_means.cluster_assignment_matrix[:, k])[0]
            print(idx.shape)
            self.clusters_covariances[k] = np.dot(data[idx].T, data[idx]) / idx.shape[0]

        self.clustered_data = np.zeros((data.shape[0], data.shape[1] + 1))
        self.cluster_probability_matrix = np.zeros((self.data.shape[0], cluster_number))

    def fit(self):
        prob_matrix = self._compute_probability_matrix(input_data=self.data,
                                                       clusters_means=self.clusters_means,
                                                       clusters_priors=self.clusters_priors,
                                                       clusters_covariances=self.clusters_covariances,
                                                       cluster_probability_matrix=self.cluster_probability_matrix)
        means = self._compute_clusters_means(prob_matrix, self.data)
        variances = self._compute_clusters_covariances(cluster_probability_matrix=prob_matrix,
                                                       cluster_means=means,
                                                       input_data=self.data)
        priors = self._compute_clusters_priors(cluster_probability_matrix=prob_matrix)

        # stop_condition - check if new covariance matrix and mi are different than old
        # define some epsilon
        # subtract two covariances matrices and sum all values check if it |v| - e > 0

        # self._initialize_cluster_centers()
        # cluster_assignment_matrix = self._compute_assignment_matrix()
        # self._compute_cluster_data_means()
        # new_cluster_assignment_matrix = self._compute_assignment_matrix()
        #
        # while np.array_equal(cluster_assignment_matrix, new_cluster_assignment_matrix) is False:
        #     cluster_assignment_matrix = new_cluster_assignment_matrix
        #     self._compute_cluster_data_means()
        #     new_cluster_assignment_matrix = self._compute_assignment_matrix()
        #     print(F'cluster_centers: \n {self.cluster_data_means}')

        max_elements_i = np.expand_dims(np.argmax(self.cluster_probability_matrix, axis=1), axis=1)
        cluster_assignment_matrix = np.zeros((self.data.shape[0], self.cluster_number))
        np.put_along_axis(cluster_assignment_matrix, max_elements_i, 1, axis=1)
        assigned_clusters = np.nonzero(cluster_assignment_matrix)[1]
        self.clustered_data[:, 0:self.data.shape[1]] = self.data
        self.clustered_data[:, self.data.shape[1]] = assigned_clusters
        print('model fitted')

    def visualize(self):
        df = pd.DataFrame(self.clustered_data)
        df.columns = ['x', 'y', 'clusters']
        sns.lmplot(data=df, x='x', y='y',
                   hue='clusters', fit_reg=False,
                   legend=False)
        plt.show()

    def _compute_clusters_sizes(self, cluster_probability_matrix):
        return np.sum(cluster_probability_matrix, axis=0)

    def _compute_clusters_means(self, cluster_probability_matrix, input_data):
        # create tensor of input data 3 x input_data.shape and then *
        return (np.dot(input_data.T, cluster_probability_matrix)).T \
               / self._compute_clusters_sizes(cluster_probability_matrix)[:, None]

    def _compute_clusters_covariances(self,
                                      cluster_probability_matrix,
                                      cluster_means,
                                      input_data):
        clusters_covariances = np.zeros((self.cluster_number, input_data.shape[1], input_data.shape[1]))
        for k in range(self.cluster_number):
            clusters_covariances[k] = self._compute_cluster_covariance(cluster_probability_matrix[k],
                                                                       cluster_means[k],
                                                                       input_data)
        return clusters_covariances

    def _compute_cluster_covariance(self,
                                    k_cluster_probability_vector,
                                    k_cluster_mean_vector,
                                    input_data):
        data_len = input_data.shape[0]
        cov_per_sample = np.zeros((data_len, input_data.shape[1], input_data.shape[1]))
        data = input_data - k_cluster_mean_vector
        for i in range(data_len):
            cov_per_sample[i] = np.dot(data[i].T, data[i])
        assert cov_per_sample[0].shape == (input_data.shape[1], input_data.shape[1])
        # k_cluster_probability_vector must be vertical
        assert k_cluster_probability_vector.shape == (data_len,)
        mul_cov_per_sample = cov_per_sample * k_cluster_probability_vector[:, None, None]
        sum_mul_cov_per_sample = np.sum(mul_cov_per_sample, axis=0)
        clusters_covariances = sum_mul_cov_per_sample / self._compute_clusters_sizes(k_cluster_probability_vector)
        return clusters_covariances

    def _compute_clusters_priors(self, cluster_probability_matrix):
        clusters_sizes = self._compute_clusters_sizes(cluster_probability_matrix)
        assert clusters_sizes.shape == (self.cluster_number, )
        clusters_priors = clusters_sizes / np.sum(clusters_sizes, axis=0)
        return clusters_priors

    def _compute_probability_matrix(self,
                                    input_data,
                                    clusters_means,
                                    clusters_covariances,
                                    clusters_priors,
                                    cluster_probability_matrix):

        def compute_gaussian_2d(input_data, cluster_mean, cluster_sigma):
            gaussian_pdf_values = np.zeros((input_data.shape[0], ))
            gaussian_pdf_values = scipy_stats.multivariate_normal.pdf(input_data, cluster_mean, cluster_sigma)
            return gaussian_pdf_values

        for k in range(self.cluster_number):
            gaussian_values = compute_gaussian_2d(input_data, clusters_means[k], clusters_covariances[k])
            cluster_probability_matrix[:, k] = clusters_priors[k] * gaussian_values
        normalize_values = np.zeros((input_data.shape[0], ))
        normalize_values = np.sum(cluster_probability_matrix, axis=1)
        cluster_probability_matrix = cluster_probability_matrix / normalize_values[:, None]
        return cluster_probability_matrix



    # Tensor algebra
    # @staticmethod
    # def _compute_covariance_2(cluster_probability_matrix, input_data):
    #     input_data = np.dot(input_data - self.clusters_data_means[k],
    #                         (self.data - self.clusters_data_means[k]).T)
    #     return GaussianMixtureModel._compute_mean_2(cluster_probability_matrix, input_data)

def main():
    data = np.genfromtxt('mickey_mouse.csv', delimiter=',')
    cluster_number = 3
    gmm = GaussianMixtureModel(cluster_number=cluster_number, data=data)
    print(gmm.cluster_number)
    print(gmm.clusters_data_means)
    print(gmm.clusters_priors)

if __name__ == '__main__':
    main()