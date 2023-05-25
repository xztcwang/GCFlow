import numpy as np
import matplotlib.pyplot as plt
import os.path
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
from oil.utils.utils import Expression, export, Named
import pickle
import subprocess
from sklearn.decomposition import PCA
from .normalization import fetch_normalization
from .sample import Sampler
from flow_ssl.data.graph_utils import sparse_mx_to_torch_sparse_tensor
from numpy import mat


def _preprocess_adj(normalization, adj):
    adj_normalizer = fetch_normalization(normalization)
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    return r_adj

class PHOTO(Dataset, metaclass=Named):
    num_classes = 8

    class_weights = None
    ignored_index = -100
    stratify = True
    def __init__(self, path='./_gcflow/data/graph_datasets/', part='all', remake=False):
        super().__init__()
        X = torch.load(path+'photo_x.pt')
        Y = torch.load(path+'photo_y.pt')
        edge_index = torch.load(path + 'photo_edge_index.pt')
        train_mask = torch.load(path + 'photo_train_mask.pt')
        val_mask = torch.load(path + 'photo_val_mask.pt')
        test_mask = torch.load(path + 'photo_test_mask.pt')
        remain_mask = torch.load(path + 'photo_remain_mask.pt')
        idx_train = torch.where(train_mask == True)[0]
        idx_val = torch.where(val_mask == True)[0]
        idx_test = torch.where(test_mask == True)[0]
        idx_remain = torch.where(remain_mask == True)[0]

        data_num = X.shape[0]
        edge_num = edge_index.shape[1]
        device = torch.device('cpu')
        X = X.to(device)
        Y = Y.to(device)
        edge_index = edge_index.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_val = idx_val.to(device)
        self.idx_test = idx_test.to(device)
        self.idx_remain = idx_remain.to(device)

        X_np = X.numpy()
        pca = PCA(n_components=50)
        pca.fit(X_np)
        X_np_new = pca.transform(X_np)
        self.X = torch.from_numpy(X_np_new)
        #self.X = X

        adj = torch.zeros(data_num, data_num)
        for edg_id in range(edge_num):
            row = edge_index[:, edg_id][0].item()
            col = edge_index[:, edg_id][1].item()
            adj[row, col] = 1
        adj = adj.detach().cpu().numpy()
        adj = mat(adj)
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        normalization = "AugRWalk"
        # normalization = "INorm"
        adj = _preprocess_adj(normalization, adj)
        self.train_adj = adj
        #self.train_adj = torch.eye(adj.shape[0]).to_sparse()

        if part == 'train':
            self.indeces_use = self.idx_train
            self.X = self.X[self.indeces_use, :]
            self.Y = Y[self.indeces_use]
        elif part == 'val':
            self.indeces_use = self.idx_val
            self.X = self.X[self.indeces_use, :]
            self.Y = Y[self.indeces_use]
        elif part == 'remain':
            self.indeces_use = self.idx_remain
            self.X = self.X[self.indeces_use, :]
            self.Y = Y[self.indeces_use]
        elif part == 'test':
            self.indeces_use = self.idx_test
            self.X = self.X[self.indeces_use, :]
            self.Y = Y[self.indeces_use]
        elif part == '_unlab':
            indeces_use = torch.cat([self.idx_val,
                                     self.idx_remain,
                                     self.idx_test],dim=0).long()
            self.indeces_use, _ = torch.sort(indeces_use, dim=0)
            self.X = self.X[self.indeces_use, :]
            self.Y = Y[self.indeces_use]
        elif part == 'all':
            self.indeces_use = torch.arange(0,data_num)
            self.X = self.X
            self.Y = Y
        else:
            print("Error Selecting The Graph Dataset")
            return
        self.dim = self.X.shape[1]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    def __len__(self):
        return self.X.shape[0]

