import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set font size of labels on matplotlib plots
plt.rc('font', size=16)

# set style of plots
sns.set_style('white')

# define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)


class KMeans:
    def __init__(self, cluster_number: int, data: np.ndarray):
        self.cluster_number = cluster_number
        self.data = data
        self.clustered_data = np.zeros((data.shape[0], data.shape[1] + 1))
        self.cluster_data_means = np.zeros((cluster_number, self.data.shape[1]))
        self.cluster_assignment_matrix = np.zeros((self.data.shape[0], cluster_number))

    def fit(self):
        self._initialize_cluster_centers()
        cluster_assignment_matrix = self._compute_assignment_matrix()
        self._compute_cluster_data_means()
        new_cluster_assignment_matrix = self._compute_assignment_matrix()

        while np.array_equal(cluster_assignment_matrix, new_cluster_assignment_matrix) is False:
            cluster_assignment_matrix = new_cluster_assignment_matrix
            self._compute_cluster_data_means()
            new_cluster_assignment_matrix = self._compute_assignment_matrix()
            print(F'cluster_centers: \n {self.cluster_data_means}')

        assigned_clusters = np.nonzero(new_cluster_assignment_matrix)[1]
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

    def _initialize_cluster_centers(self):
        self.cluster_data_means = self.data[np.random.choice(self.data.shape[0], self.cluster_number, replace=False)]

    def _compute_assignment_matrix(self):
        distance_matrix = np.copy(self.cluster_assignment_matrix)
        for i in range(self.cluster_number):
            distance_matrix[:, i] = self._euclidean_distance(self.cluster_data_means[i])
        max_elements_i = np.expand_dims(np.argmin(distance_matrix, axis=1), axis=1)
        self.cluster_assignment_matrix = np.zeros((self.data.shape[0], self.cluster_number))
        np.put_along_axis(self.cluster_assignment_matrix, max_elements_i, 1, axis=1)
        return self.cluster_assignment_matrix

    def _euclidean_distance(self, cluster_center: np.ndarray) -> np.ndarray:
        assert self.data.shape[1] == cluster_center.shape[0]
        return np.sqrt(np.sum(np.square(self.data - cluster_center), axis=1))

    def _compute_cluster_data_means(self):
        for i in range(self.cluster_number):
            self.cluster_data_means[i] = self._compute_mean(self.cluster_assignment_matrix[:, i])

    def _compute_mean(self, n_cluster_assignment_vector: np.ndarray) -> float:
        idx = np.nonzero(n_cluster_assignment_vector)[0]
        return np.sum(self.data[idx], axis=0) / idx.shape[0]


def main():
    k_means = KMeans(data=np.genfromtxt('mickey_mouse.csv', delimiter=','),
                     cluster_number=4
                     )
    k_means.fit()
    k_means.visualize()


if __name__ == '__main__':
    main()
