import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets



def plot_tsne(Z, labels, centroids,step=None):
    #n_samples, n_features = Z.shape
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    Z_tsne = tsne.fit_transform(Z)
    z_min, z_max = Z_tsne.min(0), Z_tsne.max(0)
    Z_norm = (Z_tsne - z_min) / (z_max - z_min)

    centroids_tsne = tsne.fit_transform(centroids)
    centroids_min, centroids_max = centroids_tsne.min(0), centroids_tsne.max(0)
    centroids_norm = (centroids_tsne - centroids_min) / (centroids_max - centroids_min)

    for i in range(Z_norm.shape[0]):
        plt.scatter(Z_norm[i, 0], Z_norm[i, 1],
                    color=plt.cm.Set1(labels[i]),s=3)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("save/tSNE_flowgmm_cora.png")
