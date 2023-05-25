import os
import sys
from flow_ssl.data import CORA,CITESEER,PUBMED,CORA_SPLIT,COMPUTERS,PHOTO,WIKICS
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam,AdamW
from oil.utils.utils import LoaderTo, cosLr, islice, dmap, FixedNumpySeed
from oil.datasetup.datasets import split_dataset
from functools import partial
from oil.tuning.args import argupdated_config
import copy
import flow_ssl.data as tabular_datasets

#import train_semisup_flowgmm_tabular as flows
import train_semisup_flowgmm_graph as graphflows
import oil.model_trainers as trainers
from oil.utils.mytqdm import tqdm


def makeTrainer(*, seed=42, gpu=0, dataset=PUBMED, pcadim=50,
                alg='flowgmm',
                network=graphflows.NSFGraphWPrior,
                num_epochs=15,
                inner_epochs=1,
                bs=19717,#bs=11701 for wikics, 2708, for cora, 3327 for citeseer, 19717 for pubmed, 13381 for computers, 7487 for photo
                lr=1e-3, optim=AdamW, trainer=graphflows.SemiFlowNSF,
                split=None,
                base_transform_type='affine-coupling',
                linear_transform_type='svd',
                net_config={'hidden_dim':256,
                            'flow_layers':12,
                            'tail_bound':4,
                            'num_bins':8,
                            'num_transform_blocks':4,
                            'use_batch_norm':0,
                            'dropout_ratio':0.1,
                            'apply_unconditional_transform':1},
                gauss_config={'means_r':1.6, 'cov_std': 1.1},
                opt_config={'weight_decay':0.0005},
                trainer_config={'log_dir':os.path.expanduser('~/tb-experiments/graph/'),
                                'log_args':{'minPeriod':.1, 'timeFrac':3/10},
                                'grad_norm_clip_value':50,
                                'unlab_weight':0.4},
                save=False):
    # Prep the datasets splits, model, and dataloaders
    if split is None:
        if dataset==CORA:
            split = {'train': 140, 'val': 500, 'remain': 1068, 'test': 1000}
        if dataset==CITESEER:
            split = {'train': 120, 'val': 500, 'remain': 1692, 'test': 1000}
        if dataset == PUBMED:
            split = {'train': 60, 'val': 500, 'remain': 18157, 'test': 1000}
        if dataset == COMPUTERS:
            split = {'train': 200, 'val': 1300, 'remain': 10881, 'test': 1000}
        if dataset == PHOTO:
            split = {'train': 80, 'val': 620, 'remain': 5787, 'test': 1000}
        if dataset == WIKICS:
            split = {'train': 580, 'val': 1769, 'remain': 3505, 'test': 5847}

    with FixedNumpySeed(0):
        datasets = split_dataset(dataset(alg=alg, pcadim=pcadim), splits=split)
        datasets['train'] = dataset(part='train',alg=alg, pcadim=pcadim)
        datasets['val'] = dataset(part='val',alg=alg, pcadim=pcadim)
        datasets['remain'] = dataset(part='remain',alg=alg, pcadim=pcadim)
        datasets['test'] = dataset(part='test',alg=alg, pcadim=pcadim)
        datasets['_unlab'] = dataset(part='_unlab',alg=alg, pcadim=pcadim)
        datasets['all'] = dataset(part='all',alg=alg, pcadim=pcadim)

        datasets['_unlab'] = dmap(lambda mb: mb[0], datasets['_unlab'])
        datasets['all'] = dmap(lambda mb: mb[0], datasets['all'])

    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    model = network(base_transform_type=base_transform_type,
                    linear_transform_type=linear_transform_type,
                    num_classes=datasets['train'].num_classes,
                    dim_in=datasets['train'].dim,
                    device=device,
                    trainloader=datasets['all'],
                    **net_config, **gauss_config).to(device)
    dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, len(datasets[k])), shuffle=(k == 'train'),
                                          num_workers=0, pin_memory=False), device) for k, v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)  # lambda e:1#
    return trainer(model, device, dataloaders, opt_constr, lr_sched, dataset, seed, **trainer_config)

if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    cfg = argupdated_config(defaults, namespace=(tabular_datasets, graphflows, trainers))
    trainer = makeTrainer(**cfg)
    trainer.dynamic_train_nsf(cfg['num_epochs'], cfg['inner_epochs'])
    print("=================\n")
    print("\n acc: {:.3f}, silhouette: {:.3f}\n".format(trainer.bestacc, trainer.SScore))



    



