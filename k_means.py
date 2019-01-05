import numpy as np


def k_means(cluster_number: int, data: np.ndarray) -> np.ndarray:
    initial_cluster_data_means = initialize_cluster_centers(cluster_number, data)
    cluster_assignment_matrix = compute_assignment_matrix(data, initial_cluster_data_means)
    cluster_data_means = compute_cluster_data_means(data, cluster_assignment_matrix)
    new_cluster_assignment_matrix = compute_assignment_matrix(data, cluster_data_means)

    while np.array_equal(cluster_assignment_matrix, new_cluster_assignment_matrix) is False:
        cluster_assignment_matrix = new_cluster_assignment_matrix
        cluster_data_means = compute_cluster_data_means(data, cluster_assignment_matrix)
        print(F'cluster_centers: \n {cluster_data_means}')
        new_cluster_assignment_matrix = compute_assignment_matrix(data, cluster_data_means)

    assigned_clusters = np.nonzero(new_cluster_assignment_matrix)[1]
    clustered_data = np.zeros((data.shape[0], data.shape[1] + 1))
    clustered_data[:, 0:data.shape[1]] = data
    clustered_data[:, data.shape[1]] = assigned_clusters
    print('algorithm_end')
    return clustered_data


def initialize_cluster_centers(cluster_number: int, data: np.ndarray) ->np.ndarray:
    return data[np.random.choice(data.shape[0], cluster_number, replace=False)]


def compute_assignment_matrix(data: np.ndarray, cluster_data_means: np.ndarray):
    cluster_number = cluster_data_means.shape[0]
    cluster_assignment_matrix = np.zeros((data.shape[0], cluster_number))
    distance_matrix = np.copy(cluster_assignment_matrix)
    for i in range(cluster_number):
        distance_matrix[:, i] = euclidean_distance(data, cluster_data_means[i])
    max_elements_i = np.expand_dims(np.argmin(distance_matrix, axis=1), axis=1)
    np.put_along_axis(cluster_assignment_matrix, max_elements_i, 1, axis=1)
    return cluster_assignment_matrix


def euclidean_distance(points: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    assert points.shape[1] == ref_point.shape[0]
    return np.sqrt(np.sum(np.square(points - ref_point), axis=1))


def compute_cluster_data_means(data: np.ndarray, cluster_assignment_matrix: np.ndarray) -> np.ndarray:
    cluster_number = cluster_assignment_matrix.shape[1]
    cluster_data_means = np.zeros((cluster_number, data.shape[1]))
    for i in range(cluster_number):
        cluster_data_means[i] = compute_mean(data, cluster_assignment_matrix[:, i])
    return cluster_data_means


def compute_mean(data: np.ndarray, n_cluster_assignment_vector: np.ndarray) -> float:
    idx = np.nonzero(n_cluster_assignment_vector)[0]
    return np.sum(data[idx], axis=0) / idx.shape[0]


def main():
    cluster_numbers = 10
    data = np.genfromtxt('mickey_mouse.csv', delimiter=',')
    clustered_data = k_means(cluster_numbers, data)
    np.savetxt("clustered_data.csv", clustered_data, delimiter=",")


if __name__ == '__main__':
    main()
