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

def beta_postprocessing(emb):
    mean = np.mean(emb, axis=0, keepdims=True)
    centred_emb = emb - mean
    u, s, vh = np.linalg.svd(centred_emb,full_matrices=False)

    def objective(b):
        l = s**2
        log1 = np.log(l)
        l2b = 1 ** (2*b)
        first_term = np.mean(log1*l2b*log1)*np.mean(l2b)
        second_term = np.mean(log1*l2b)**2
        derivative = -1./(np.mean(l2b)**2.)*(first_term-second_term)
        return derivative
    values = list(map(objective,np.arange(0.5, 1.0, 0.001)))
    optimal_beta = np.argmin(values)/1000. + 0.5
    processed_emb = u@np.diag(s**optimal_beta)@vh
    return processed_emb

class CORA_SPLIT(Dataset, metaclass=Named):
    num_classes = 7
    class_weights = None
    ignored_index = -100
    stratify = True

    #def __init__(self, root='~/datasets/UCI/miniboone/', train=True, remake=False):
    def __init__(self, path='./_gcflow/data/graph_datasets', part='all', remake=False, ptscls=None):
        super().__init__()
        sampler = Sampler("cora", path, "semi")
        # 'NormLap': normalized_laplacian,  # A' = I - D^-1/2 * A * D^-1/2
        # 'Lap': laplacian,  # A' = D - A
        # 'RWalkLap': random_walk_laplacian,  # A' = I - D^-1 * A
        # 'FirstOrderGCN': gcn,  # A' = I + D^-1/2 * A * D^-1/2
        # 'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        # 'BingGeNormAdj': bingge_norm_adjacency,  # A' = I + (D + I)^-1/2 * (A + I) * (D + I)^-1/2
        # 'NormAdj': normalized_adjacency,  # D^-1/2 * A * D^-1/2
        # 'RWalk': random_walk,  # A' = D^-1*A
        # 'AugRWalk': aug_random_walk,  # A' = (D + I)^-1*(A + I)
        # 'NoNorm': no_norm,  # A' = A
        # 'INorm': i_norm,  # A' = A + I
        device = torch.device('cpu')
        labels, idx_train, idx_val, idx_remain, idx_test = sampler.get_label_and_idxes(device)
        train_adj, train_fea = sampler.randomedge_sampler(percent=1.0,
                                                          #normalization="AugNormAdj",
                                                          normalization="AugRWalk",
                                                          cuda=device)
        ##splitting:
        if ptscls is not None:
            num_perclass = ptscls
            idx_pool = torch.cat((idx_train, idx_remain), dim=0)
            labeled_idx_sampled = [torch.Tensor([]) for _ in range(self.num_classes)]
            for idx in idx_pool:
                point_cls = labels[idx]
                idx_tensor = torch.Tensor([idx]).long()
                if labeled_idx_sampled[point_cls].shape[0] < num_perclass:
                    labeled_idx_sampled[point_cls] = torch.cat((labeled_idx_sampled[point_cls], idx_tensor),
                                                               dim=0).long()

            labeled_idx_tensor = torch.Tensor([])
            for cls in range(self.num_classes):
                labeled_idx_tensor = torch.cat((labeled_idx_tensor, labeled_idx_sampled[cls]),
                                               dim=0).long()
            sorted_labeled_idx_tensor = labeled_idx_tensor.sort(0, False).values
            sorted_labeled_idx_list = sorted_labeled_idx_tensor.detach().numpy().tolist()
            idx_pool_list = idx_pool.detach().numpy().tolist()

            idx_remain_list = list(set(idx_pool_list).difference(set(sorted_labeled_idx_list)))
            idx_remain_new = torch.Tensor(idx_remain_list).long()
            idx_train_new = sorted_labeled_idx_tensor
            idx_train = idx_train_new
            idx_remain = idx_remain_new

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_remain = idx_remain
        self.idx_test = idx_test
        #self.train_adj = train_adj
        self.train_adj = torch.eye(train_adj.shape[0]).to_sparse()
        #self.train_adj = torch.eye(train_adj.shape[0]).to_sparse()


        #self.X = train_fea
        train_fea_np=train_fea.numpy()
        pca = PCA(n_components=50)
        pca.fit(train_fea_np)
        train_fea_np_new = pca.transform(train_fea_np)
        # processed_emb = beta_postprocessing(train_fea_np_new)

        # self.X = torch.from_numpy(processed_emb)
        self.X=torch.from_numpy(train_fea_np_new)

        #class_number = labels.max().item()+1
        #train_number = len(idx_train)




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
                                          self.idx_test],dim=0)
            self.X = self.X[self.indeces_use, :]
            self.Y = labels[self.indeces_use]
        elif part == 'all':
            self.indeces_use = torch.cat([self.idx_train,
                                          self.idx_val,
                                          self.idx_remain,
                                          self.idx_test], dim=0)
            #self.X = self.X[self.indeces_use, :]
            #self.Y = labels[self.indeces_use]
            self.X = self.X
            self.Y = labels
        else:
            print("Error Selecting The Graph Dataset")
            return

        self.dim = self.X.shape[1]


        #self.X = torch.from_numpy(self.X_train if train else self.X_test).float()
        #self.Y = torch.from_numpy(self.y_train if train else self.y_test).long()




    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.X.shape[0]

    # def show_histograms(self, split, vars):

    #     data_split = getattr(self, split, None)
    #     if data_split is None:
    #         raise ValueError('Invalid data split')

    #     util.plot_hist_marginals(data_split.x[:, vars])
    #     plt.show()


def load_data(root_path):
    # NOTE: To remember how the pre-processing was done.
    # print("got here")
    data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
    nsignal = int(data.iloc[0][0])
    nbackground = int(data.iloc[0][1])
    print("{} signal, {} background".format(nsignal, nbackground))
    minimum = min(nsignal, nbackground)
    labels = np.concatenate((np.ones(minimum), np.zeros(minimum)))
    data = data.iloc[1:].values
    data = np.concatenate((data[:minimum], data[nsignal:nsignal + minimum]))
    # print("got here")
    # Remove some random outliers
    # indices = (data[:, 0] < -100)
    # data = data[~indices]
    # labels = labels[~indices]
    i = 0
    # Remove any features that have too many re-occuring real values.
    # features_to_remove = []
    # for feature in data.T:
    #     c = Counter(feature)
    #     max_count = np.array([v for k, v in sorted(c.items())])[0]
    #     if max_count > 5:
    #         features_to_remove.append(i)
    #     i += 1
    # print(features_to_remove)
    # print(np.array([i for i in range(data.shape[1]) if i not in features_to_remove]))
    # print(data.shape)
    # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
    # np.save("~/data/miniboone/data.npy", data)
    return data, labels


def load_data_normalised(root_path):
    data, labels = load_data(root_path)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data, labels
