import torch
from torch.utils.data import Dataset
from oil.utils.utils import Named
from sklearn.decomposition import PCA
from .sample import Sampler



class CORA(Dataset, metaclass=Named):
    num_classes = 7
    class_weights = None
    ignored_index = -100
    stratify = True

    def __init__(self, path='../../data/graph_datasets', part='all',
                 alg='gcflow', pcadim=50, remake=False):
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
                                                          normalization="AugRWalk",
                                                          cuda=device)
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_remain = idx_remain
        self.idx_test = idx_test
        self.idx_all = torch.arange(start=0, end=len(labels), step=1)
        if alg=='gcflow':
            self.train_adj = train_adj
        if alg=='flowgmm':
            self.train_adj = torch.eye(train_adj.shape[0]).to_sparse()

        #self.X = train_fea
        train_fea_np=train_fea.numpy()
        pca = PCA(n_components=pcadim)
        pca.fit(train_fea_np)
        train_fea_np_new = pca.transform(train_fea_np)
        self.X=torch.from_numpy(train_fea_np_new)


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


