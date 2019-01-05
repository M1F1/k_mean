import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        k_means = KMeans(data=np.genfromtxt('mickey_mouse.csv', delimiter=','),
                         cluster_number=4
                         )
        k_means.fit()
        self.cluster_number = cluster_number
        self.data = data
        self.clustered_data = np.zeros((data.shape[0], data.shape[1] + 1))
        self.cluster_data_means = np.zeros((cluster_number, self.data.shape[1]))
        self.cluster_assignment_matrix = np.zeros((self.data.shape[0], cluster_number))

    def fit(self):
        self._initialize_cluster_centers()
        self._compute_assignment_matrix()
        self._compute_cluster_data_means()
        new_cluster_assignment_matrix = self._compute_assignment_matrix()

        while np.array_equal(self.cluster_assignment_matrix, new_cluster_assignment_matrix) is False:
            self.cluster_assignment_matrix = new_cluster_assignment_matrix
            self._compute_cluster_data_means()
            print(F'cluster_centers: \n {self.cluster_data_means}')
            new_cluster_assignment_matrix = self._compute_assignment_matrix()

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
def gussian_mixture_model():

def main():

    gaussian_mixture_model()
if __name__ == '__main__':
    main()