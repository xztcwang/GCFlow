import torch
from torch import distributions, nn
import torch.nn.functional as F
from torch.distributions import multivariate_normal
import numpy as np
import math

class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self, trainloader, means, means_r, inv_cov_stds=None, device=None):
        self.device = device
        self.n_components, self.d = means.shape
        self.means = means.to(self.device)
        self.trainloader = trainloader

        self.data_number=self.trainloader.X.shape[0]
        self.class_number=means.shape[0]

        if inv_cov_stds is None:
            self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones((len(means)), device=device)
        else:
            self.inv_cov_stds = inv_cov_stds.to(self.device)

        self.covariance = [None for _ in range(self.n_components)]

        self.weights = torch.ones((len(means)), device=device)

        self.lab_idx = self.trainloader.idx_train
        self.unlab_idx=torch.cat([self.trainloader.idx_val,
                                  self.trainloader.idx_remain,
                                  self.trainloader.idx_test], dim=0).long()
        self.labels = self.trainloader.Y
        self.scale = means_r



        self.membership = torch.zeros([self.class_number,self.data_number], device=device)



    @property
    def gaussians(self):
        gaussians = []
        for k in range(self.n_components):
            mean_k = self.get_mean(k)
            sigma_k = self.get_sigma(k)
            gaussians.append(distributions.MultivariateNormal(mean_k,sigma_k,validate_args=False))
        #gaussians = [distributions.MultivariateNormal(mean, F.softplus(inv_std)**2 * torch.eye(self.d).to(self.device))
        #                  for mean, inv_std in zip(self.means, self.inv_cov_stds)]
        return gaussians

    def get_mean(self,k):
        mean_k = self.means[k,:]
        return mean_k

    def get_sigma(self,k):
        if self.covariance[k] is None:
            sigma_k = F.softplus(self.inv_cov_stds[k]) ** 2 * torch.eye(self.d).to(self.device)
        else:
            sigma_k = self.covariance[k]
        if not torch.all(sigma_k.transpose(0, 1) == sigma_k).item():
            print("not symmetric")
        if not bool((torch.eig(sigma_k)[0][:, 0] >= 0).all()):
            print("negative eig")
        return sigma_k

    def parameters(self):
       return [self.means, self.inv_cov_stds, self.weights]
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)[0]
                samples[mask] = all_samples[i][mask]
        return samples
        
    def log_prob(self, x, y=None, label_weight=1.):
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights,dim=0)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                #Pavel: add class weights here? 
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights,dim=0))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)

    def update_membership(self, z):

        for data_idx in self.lab_idx:
            label = self.labels[data_idx]
            self.membership[label, data_idx] = 1

        if self.covariance[0] is None:
            dist = [distributions.MultivariateNormal(mean, F.softplus(inv_std) ** 2 * torch.eye(self.d).to(self.device))
                              for mean, inv_std in zip(self.means, self.inv_cov_stds)]
        else:
            dist = [distributions.MultivariateNormal(mean, covar)
                    for mean, covar in zip(self.means, self.covariance)]

        for data_idx in self.unlab_idx:
            zx = z[data_idx, :]
            pdfs = []
            for k in range(self.n_components):
                pdf_k = dist[k].log_prob(zx) + self.weights[k].log()
                # mean_k = self.get_mean(k)
                # sigma_k = self.get_sigma(k)
                # dist_k = multivariate_normal.MultivariateNormal(loc=mean_k, covariance_matrix=sigma_k)
                # pdf_k = dist_k.log_prob(zx) + self.weights[k].log()
                pdfs.append(pdf_k)
            norm_pdfs = F.softmax(torch.Tensor(pdfs),dim=0)
            for k in range(self.n_components):
                self.membership[k, data_idx] = norm_pdfs[k].item()

    def update_covariance(self, z):
        for k in range(self.n_components):
            self.covariance[k] = torch.zeros(self.d, self.d).to(self.device)
            #z=z.type(torch.float64)
            #self.membership = self.membership.type(torch.float64)
            zmu = z - self.means[k]
            zmut = (z - self.means[k]).transpose(0, 1)
            wk = torch.diag(self.membership[k, :])
            zmutwk = torch.mm(zmut, wk)
            zmutzmu = torch.mm(zmutwk, zmu)
            normalizer = self.membership[k, :].sum().item()
            A = torch.div(zmutzmu, normalizer)
            A_ = 0.5*(A+A.transpose(0, 1))
            max_eig = torch.eig(A_)[0][:, 0].max()
            self.covariance[k] = A_ + 1e-4 * max_eig * torch.eye(self.d).to(self.device)
            if not torch.all(self.covariance[k].transpose(0, 1) == self.covariance[k]).item():
                print("not symmetric")
            if not bool((torch.eig(self.covariance[k])[0][:,0]>=0).all()):
                print("negative eig")



        # for k in range(self.n_components):
        #     self.covariance[k] = torch.zeros(self.d, self.d).to(self.device)
        #     zmu = z-self.means[k]
        #     zmut = (z-self.means[k]).transpose(0,1)
        #     wk = torch.diag(self.membership[k, :])
        #     zmutwk = torch.mm(zmut, wk)
        #     zmutzmu = torch.mm(zmutwk, zmu)
        #     #zmutzmu = zmutzmu + 1e-2 * torch.eye(self.d).to(self.device)
        #     normalizer = self.membership[k, :].sum().item()
        #     #self.covariance[k] = torch.div(zmutzmu, normalizer)
        #     no_damping = torch.div(zmutzmu, normalizer)
        #     max_eig = torch.eig(no_damping)[0][:, 0].max()
        #     self.covariance[k] = no_damping + 1e-6 * max_eig * torch.eye(self.d).to(self.device)
        #     if not torch.all(self.covariance[k].transpose(0, 1) == self.covariance[k]).item():
        #         print("not symmetric")

        # for k in range(self.n_components):
        #     self.covariance[k] = torch.zeros(self.d, self.d).to(self.device)
        #     for data_id in range(self.data_number):
        #         zmu = z[data_id, :] - self.means[k]
        #         zmut = torch.unsqueeze(zmu, dim=1)
        #         self.covariance[k] += self.membership[k, data_id] * torch.mul(zmut, zmu)
        #     normalizer = self.membership[k, :].sum().item()
        #     self.covariance[k] = torch.div(self.covariance[k], normalizer)
        #     max_eig = torch.eig(self.covariance[k])[0][:, 0].max()
        #     self.covariance[k] = self.covariance[k] + 1e-6 * max_eig * torch.eye(self.d).to(self.device)
        #     if not torch.all(self.covariance[k].transpose(0, 1) == self.covariance[k]).item():
        #         print("not symmetric")




    def update_means(self, z):
        for k in range(self.n_components):
            w_k = self.membership[k, :].reshape(1, self.data_number)
            normalizer = w_k.sum()
            mean = torch.mm(w_k, z)/normalizer
            self.means[k] = mean.reshape(self.d)

        # for k in range(self.n_components):
        #     indeces = self.labels[self.lab_idx]==k
        #     self.means[k] += z[self.lab_idx == k, :].sum(dim=0)
        #     self.means[k] /= sum(self.lab_idx == k)
        #     self.means[k] = self.means[k] * self.scale
        #     self.means[k] = self.means[k].reshape(self.d)

    def update_weights(self):
        for k in range(self.n_components):
            self.weights[k] = (self.membership[k, :].sum()/self.data_number).item()







#PAVEL: remove later
class SSLGaussMixtureClassifier(SSLGaussMixture):
    
    def __init__(self, means, cov_std=1., device=None):
        super().__init__(means, cov_std, device)
        self.classifier = nn.Sequential(nn.Linear(self.d, self.n_components))

    def parameters(self):
       return self.classifier.parameters() 

    def forward(self, x):
        return self.classifier.forward(x)

    def log_prob(self, x, y, label_weight=1.):
        all_probs = [torch.exp(g.log_prob(x)) for g in self.gaussians]
        probs = sum(all_probs) / self.n_components
        x_logprobs = torch.log(probs)

        mask = (y != -1)
        labeled_x, labeled_y = x[mask], y[mask].long()
        preds = self.forward(labeled_x)
        y_logprobs = F.cross_entropy(preds, labeled_y)

        return x_logprobs - y_logprobs
