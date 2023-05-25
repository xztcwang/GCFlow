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


from .sample import Sampler

class CITESEER(Dataset, metaclass=Named):
    num_classes = 6
    class_weights = None
    ignored_index = -100
    stratify = True

    def __init__(self, path='../../data/graph_datasets', part='all',
                 alg='gcflow', pcadim=50, remake=False):
        super().__init__()
        sampler = Sampler("citeseer", path, "semi")
        device = torch.device('cpu')
        labels, idx_train, idx_val, idx_remain, idx_test = sampler.get_label_and_idxes(device)
        train_adj, train_fea = sampler.randomedge_sampler(percent=1.0,
                                                          normalization="AugRWalk",
                                                          cuda=device)
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_remain = idx_remain
        self.idx_test = idx_test
        self.idx_all = torch.arange(start=0, end=len(labels), step=1)
        if alg == 'gcflow':
            self.train_adj = train_adj
        if alg == 'flowgmm':
            self.train_adj = torch.eye(train_adj.shape[0]).to_sparse()

        # self.X = train_fea
        train_fea_np = train_fea.numpy()
        pca = PCA(n_components=pcadim)
        pca.fit(train_fea_np)
        train_fea_np_new = pca.transform(train_fea_np)
        self.X = torch.from_numpy(train_fea_np_new)

        if part == 'train':
            self.indeces_use = self.idx_train
            self.X = self.X[self.indeces_use, :]
            self.Y = labels[self.indeces_use]
        elif part == 'val':
            self.indeces_use = self.idx_val
            self.X = self.X[self.indeces_use, :]
            self.Y = labels[self.indeces_use]
        elif part == 'remain':
            self.indeces_use = self.idx_remain
            self.X = self.X[self.indeces_use, :]
            self.Y = labels[self.indeces_use]
        elif part == 'test':
            self.indeces_use = self.idx_test
            self.X = self.X[self.indeces_use, :]
            self.Y = labels[self.indeces_use]
        elif part == '_unlab':
            self.indeces_use = torch.cat([self.idx_val,
                                          self.idx_remain,
                                          self.idx_test], dim=0)
            # self.indeces_use = torch.arange(start=self.idx_train.max().item() + 1,
            #                                 end=self.idx_all.max().item() + 1, step=1)
            self.X = self.X[self.indeces_use, :]
            self.Y = labels[self.indeces_use]
        elif part == 'all':
            self.indeces_use = self.idx_all
            self.X = self.X
            self.Y = labels
        else:
            print("Error Selecting The Graph Dataset")
            return

        self.dim = self.X.shape[1]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.X.shape[0]
