import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from oil.model_trainers.classifier import Trainer
from oil.utils.utils import Eval, izip, icycle, export
import utils
from flow_ssl.distributions import SSLGaussMixture
from flow_ssl.tsne import plot_tsne
from scipy.spatial.distance import cdist
from flow_ssl.nde import distributions, flows, transforms
import flow_ssl.nsf_utils as nsf_utils
import flow_ssl.nn as nn_
from sklearn.metrics import f1_score
import flow_ssl.nde.structures as graphstrucres


def create_linear_transform(linear_transform_type, dim_in, dropout_ratio, device, flowlayer):
    if linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=dim_in)
    elif linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=dim_in),
            transforms.LULinear(dim_in, identity_init=True)
        ], dropout_ratio, device, flowlayer)
    elif linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=dim_in),
            transforms.SVDLinear(dim_in, num_householder=10)
        ], dropout_ratio, device, flowlayer)
    else:
        raise ValueError

def create_base_transform(i,
                          base_transform_type,
                          dim_in,
                          hidden_dim,
                          num_transform_blocks,
                          tail_bound,
                          num_bins,
                          dropout_ratio,
                          use_batch_norm,
                          apply_unconditional_transform):
    if base_transform_type == 'affine-coupling':
        return transforms.AffineCouplingTransform(
            mask=nsf_utils.create_alternating_binary_mask(dim_in, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_dim,
                context_features=None,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=dropout_ratio,
                use_batch_norm=use_batch_norm
            )
        )
    elif base_transform_type == 'quadratic-coupling':
        return transforms.PiecewiseQuadraticCouplingTransform(
            mask=nsf_utils.create_alternating_binary_mask(dim_in, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_dim,
                context_features=None,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=dropout_ratio,
                use_batch_norm=use_batch_norm
            ),
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform
        )
    elif base_transform_type == 'rq-coupling':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=nsf_utils.create_alternating_binary_mask(dim_in, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_dim,
                context_features=None,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=dropout_ratio,
                use_batch_norm=use_batch_norm
            ),
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform
        )
    elif base_transform_type == 'affine-autoregressive':
        return transforms.MaskedAffineAutoregressiveTransform(
            features=dim_in,
            hidden_features=hidden_dim,
            context_features=None,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_ratio,
            use_batch_norm=use_batch_norm
        )
    elif base_transform_type == 'quadratic-autoregressive':
        return transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
            features=dim_in,
            hidden_features=hidden_dim,
            context_features=None,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_ratio,
            use_batch_norm=use_batch_norm
        )
    elif base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=dim_in,
            hidden_features=hidden_dim,
            context_features=None,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_ratio,
            use_batch_norm=use_batch_norm
        )
    else:
        raise ValueError

def create_transform(flow_layers,
                     linear_transform_type,
                     dim_in,
                     base_transform_type,
                     hidden_dim,
                     num_transform_blocks,
                     tail_bound,
                     num_bins,
                     dropout_ratio,
                     use_batch_norm,
                     apply_unconditional_transform,
                     device
                     ):
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(linear_transform_type, dim_in, dropout_ratio, device, flowlayer=False),
            create_base_transform(i,
                                  base_transform_type,
                                  dim_in,
                                  hidden_dim,
                                  num_transform_blocks,
                                  tail_bound,
                                  num_bins,
                                  dropout_ratio,
                                  use_batch_norm,
                                  apply_unconditional_transform)
        ], dropout_ratio, device, flowlayer=True) for i in range(flow_layers)
    ] + [
        create_linear_transform(linear_transform_type, dim_in, dropout_ratio, device, flowlayer=False)
    ], dropout_ratio, device, flowlayer=False)
    return transform




@export
def NSFGraphWPrior(base_transform_type, linear_transform_type, num_classes, dim_in,
                   device, flow_layers, hidden_dim, num_transform_blocks, tail_bound, num_bins,
                   dropout_ratio,use_batch_norm,apply_unconditional_transform,
                   means_r, cov_std, trainloader=None):
    # create model
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    idx_train = trainloader.idx_train

    means = utils.get_means_graph('random', r=means_r * .7, num_means=num_classes, trainloader=trainloader,
                                  train_idx=idx_train, shape=(dim_in), device=device)
    distribution = distributions.StandardNormal((dim_in,))

    transform = create_transform(flow_layers, linear_transform_type, dim_in, base_transform_type,
                                 hidden_dim, num_transform_blocks, tail_bound, num_bins,
                                 dropout_ratio, use_batch_norm, apply_unconditional_transform,
                                 device)
    model = flows.Flow(transform, distribution).to(device)
    n_params = nsf_utils.get_num_parameters(model)
    print('There are {} trainable parameters in this model.'.format(n_params))
    model.prior = SSLGaussMixture(trainloader, means, means_r * .7, inv_cov_std, device=device)
    means_np = means.detach().cpu().numpy()
    print("Pairwise dists:", cdist(means_np, means_np))

    return model


@export
class SemiFlowNSF(Trainer):
    def __init__(self, *args, unlab_weight=1.,cons_weight=3.,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'unlab_weight':unlab_weight,'cons_weight':cons_weight})
        #self.dataloaders['train'] = izip(icycle(self.dataloaders['train']),self.dataloaders['_unlab'])
        self.idx_train = self.dataloaders['all'].dataset.idx_train.to(self.device)
        self.idx_val = self.dataloaders['all'].dataset.idx_val.to(self.device)
        self.idx_test = self.dataloaders['all'].dataset.idx_test.to(self.device)
        self.idx_remain=self.dataloaders['all'].dataset.idx_remain.to(self.device)
        self.idx_all = self.dataloaders['all'].dataset.indeces_use.to(self.device)
        self.adj_mat = self.dataloaders['all'].dataset.train_adj.to(self.device)
        self.idx_unlab = torch.cat([self.idx_val,self.idx_remain,self.idx_test], dim=0).long()
        self.dataloaders['train'] = izip(icycle(self.dataloaders['train']), self.dataloaders['all'])
        self.train_labels=self.dataloaders['all'].dataset.Y[self.idx_train].to(self.device)
        self.val_labels=self.dataloaders['all'].dataset.Y[self.idx_val].to(self.device)
        self.test_labels=self.dataloaders['all'].dataset.Y[self.idx_test].to(self.device)
        self.all_labels=self.dataloaders['all'].dataset.Y.to(self.device)
        self.class_num=self.dataloaders['all'].dataset.num_classes

    def loss(self, minibatch):
        x_adj = (minibatch[1], self.adj_mat)
        a = float(self.hypers['unlab_weight'])
        b = float(self.hypers['cons_weight'])
        z, logabsdet = self.model.log_prob(x_adj=x_adj)
        prior_ll_lab = self.model.prior.log_prob(z[self.idx_train], self.train_labels)
        prior_ll_unlab = self.model.prior.log_prob(z[self.idx_unlab])
        logabsdet_lab = logabsdet[self.idx_train]
        logabsdet_unlab = logabsdet[self.idx_unlab]

        prior_ll_lab_term = -prior_ll_lab.sum()/len(self.idx_train)
        prior_ll_unlab_term = -prior_ll_unlab.sum()/len(self.idx_unlab)
        logabsdet_lab_term = -logabsdet_lab.sum()/len(self.idx_train)
        logabsdet_unlab_term = -logabsdet_unlab.sum()/len(self.idx_unlab)
        flow_loss = (1-a)*(prior_ll_lab_term + logabsdet_lab_term) + \
                    a * (prior_ll_unlab_term + logabsdet_unlab_term)

        return flow_loss

    def step(self, minibatch):
        self.optimizer.zero_grad()
        #self.scheduler.step(step)
        loss = self.loss(minibatch)
        loss.backward()
        utils.clip_grad_norm(self.optimizer, self.grad_norm_clip_value)
        self.optimizer.step()
        return loss

    def logStuff(self, step, minibatch=None):
        eval_acc_fun = lambda x_adj, idx, label: self.model.prior.classify((self.model.log_prob(x_adj=x_adj))[0][idx])\
            .type_as(label).eq(label).cpu().data.numpy().mean()
        metrics = {}
        with Eval(self.model), torch.no_grad():
            metrics['Train_Acc'] = self.evalAverageMetrics(self.dataloaders['all'], self.adj_mat, self.idx_train,
                                                           eval_acc_fun, self.train_labels)
            metrics['val_Acc'] = self.evalAverageMetrics(self.dataloaders['all'], self.adj_mat, self.idx_val,
                                                         eval_acc_fun, self.val_labels)
            metrics['test_Acc'] = self.evalAverageMetrics(self.dataloaders['all'], self.adj_mat, self.idx_test,
                                                          eval_acc_fun, self.test_labels)

            if metrics['val_Acc']>self.best_val_acc:
                self.best_val_acc=metrics['val_Acc']
                self.bestacc=metrics['test_Acc']
                self.best_epoch=step
                # mb1_adj = (minibatch[1], self.adj_mat)
                # z, logabsdet = self.model.log_prob(x_adj=mb1_adj)
                #pred = self.model.prior.classify(z).type_as(self.all_labels)
                # self.bestari, self.bestmi, _, _ = utils.clustering_measure(z, pred, self.class_num,
                #                                              self.all_labels, self.idx_train,
                #                                              self.device)

            if minibatch:
                #calculate clustering accuracy
                if step == self.totalep - 1:
                    mb1_adj = (minibatch[1], self.adj_mat)
                    z, logabsdet = self.model.log_prob(x_adj=mb1_adj)
                    pred = self.model.prior.classify(z).type_as(self.all_labels)
                    self.ARI, self.MI, self.SScore, centroids= utils.clustering_measure(z, pred, self.class_num, self.all_labels, self.idx_train, self.device)

                    #plot_tsne(Z=z.detach().cpu().numpy(), labels=self.all_labels.detach().cpu().numpy(),centroids=centroids)

        self.logger.add_scalars('metrics', metrics, step + self.inner_epoch / 10)
        super().logStuff(step + self.inner_epoch / 10, minibatch)
        if step == self.totalep - 1:
            self.finalacc = metrics['test_Acc']






