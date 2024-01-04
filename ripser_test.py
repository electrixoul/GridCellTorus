from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

data = datasets.make_circles(n_samples=1000)[0] + 5 * datasets.make_circles(n_samples=1000)[0]

data_blob = datasets.make_blobs(n_samples=1000, n_features=2, centers=1, cluster_std=0.1)[0]
data_moon = datasets.make_moons(n_samples=1000, noise=0.05)[0]

# 制造一个 torus
data_torus = np.zeros((1000, 3))
for i in range(1000):
    theta = np.random.rand() * 2 * np.pi
    phi = np.random.rand() * 2 * np.pi
    data_torus[i][0] = (2 + np.cos(theta)) * np.cos(phi)
    data_torus[i][1] = (2 + np.cos(theta)) * np.sin(phi)
    data_torus[i][2] = np.sin(theta)

# 计算 data 的自相关矩阵
data_self = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
    for j in range(data.shape[0]):
        data_self[i][j] = np.linalg.norm(data[i] - data[j])

data_torus_self = np.zeros((data_torus.shape[0], data_torus.shape[0]))
for i in range(data_torus.shape[0]):
    for j in range(data_torus.shape[0]):
        data_torus_self[i][j] = np.linalg.norm(data_torus[i] - data_torus[j])

print("shape of data: ", data_moon.shape)

# dgms = ripser(data_self, distance_matrix = True)['dgms']
# dgms = ripser(data)['dgms']
dgms = ripser(data_torus_self, distance_matrix = True)['dgms']
# dgms = ripser(data_torus)['dgms']
plot_diagrams(dgms, show=True)

# # Plot each diagram by itself
# plot_diagrams(dgms, plot_only=[0], ax=plt.subplot(121))
# plot_diagrams(dgms, plot_only=[1], ax=plt.subplot(122))

# # Homology over Z/3Z
# dgms = ripser(data, coeff=3)['dgms']
# plot_diagrams(dgms, plot_only=[1], title="Homology of Z/3Z", show=True)

# # Homology over Z/7Z
# dgms = ripser(data, coeff=3)['dgms']
# plot_diagrams(dgms, plot_only=[1], title="Homology of Z/7Z", show=True) # Only plot H_1
