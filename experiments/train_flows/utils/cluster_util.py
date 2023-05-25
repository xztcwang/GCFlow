import torch
import torch.nn.functional as F
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
#from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


def clustering_measure(Z,preds,nclass,labels,idx_train,device):
    nfeat = Z.shape[1]
    Z_means_tensor = torch.Tensor([]).to(device)
    for c in range(nclass):
        c_ind = torch.where(preds == c)[0].detach().cpu().numpy().tolist()
        if len(c_ind):
            Z_mean_c = Z[c_ind].mean(dim=0)
            Z_mean_c = Z_mean_c.reshape(1, nfeat)
            Z_means_tensor = torch.cat((Z_means_tensor, Z_mean_c), dim=0)
        else:
            Z_mean_c=torch.rand([1, nfeat]).to(device)
            Z_means_tensor = torch.cat((Z_means_tensor, Z_mean_c), dim=0)
    Z_means_init_np = Z_means_tensor.detach().cpu().numpy()


    km = KMeans(n_clusters=nclass, random_state=0, init=Z_means_init_np, n_init=nclass,
                max_iter=1000, tol=0.0001, verbose=0, copy_x=True, algorithm='auto')
    Z_np = Z.detach().cpu().numpy()
    kmfit = km.fit(Z_np)
    km_predlab = kmfit.labels_
    centroids=kmfit.cluster_centers_
    labels_np = labels.detach().cpu().numpy()
    true_labels = labels_np
    pp=preds.cpu().numpy()
    #MI = adjusted_mutual_info_score(true_labels, pp, average_method='arithmetic')
    MI = normalized_mutual_info_score(true_labels, pp)
    S_Score = metrics.silhouette_score(Z_np, km_predlab, metric='euclidean')
    ARI = adjusted_rand_score(true_labels, pp)
    return ARI, MI, S_Score, centroids






