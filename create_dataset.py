import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    gauss1 = stats.multivariate_normal([0, 0], [[20, 0], [0, 20]])
    gauss2 = stats.multivariate_normal([12, 12], [[3, 0], [0, 3]])
    gauss3 = stats.multivariate_normal([-12, 12], [[3, 0], [0, 3]])

    dataset = []
    for _ in range(600):
        p1 = gauss1.rvs()
        p1 = np.append(p1, 1)
        dataset.append(p1)
    for _ in range(200):
        p2 = gauss2.rvs()
        p2 = np.append(p2, 2)
        dataset.append(p2)
    for _ in range(200):
        p3 = gauss3.rvs()
        p3 = np.append(p3, 3)
        dataset.append(p3)

    dataset = np.array(dataset)
    np.savetxt("mickey_mouse.csv", dataset, delimiter=",")
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.show()
    # fig, ax = plt.subplots(1, 1)
    # plt.axis('equal')
    # ax.scatter(x=dataset[:,0], y=dataset[:,1])

if __name__ == '__main__':
    main()