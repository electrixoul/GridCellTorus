from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

data = datasets.make_circles(n_samples=1000)[0] + 5 * datasets.make_circles(n_samples=1000)[0]

print("shape of data: ", data.shape)

dgms = ripser(data)['dgms']
plot_diagrams(dgms, show=True)

# Plot each diagram by itself
plot_diagrams(dgms, plot_only=[0], ax=plt.subplot(121))
plot_diagrams(dgms, plot_only=[1], ax=plt.subplot(122))

# Homology over Z/3Z
dgms = ripser(data, coeff=3)['dgms']
plot_diagrams(dgms, plot_only=[1], title="Homology of Z/3Z", show=True)

# Homology over Z/7Z
dgms = ripser(data, coeff=3)['dgms']
plot_diagrams(dgms, plot_only=[1], title="Homology of Z/7Z", show=True) # Only plot H_1
