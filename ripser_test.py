from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

data = datasets.make_circles(n_samples=1000)[0] + 5 * datasets.make_circles(n_samples=1000)[0]

data_blob = datasets.make_blobs(n_samples=1000, n_features=2, centers=1, cluster_std=0.1)[0]
data_moon = datasets.make_moons(n_samples=1000, noise=0.05)[0]

# 计算 data 的自相关矩阵
data_self = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
    for j in range(data.shape[0]):
        data_self[i][j] = np.linalg.norm(data[i] - data[j])

print("shape of data: ", data_moon.shape)

dgms = ripser(data_self, distance_matrix = True)['dgms']
# dgms = ripser(data)['dgms']
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
