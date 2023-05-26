# GC-Flow
This repository contains the code for [GC-Flow: A Graph-Based Flow Network for Effective Clustering](https://openreview.net/pdf?id=NRJPnlZ1JI) publised in ICML 2023.
## Installation
To run the project you will need to install the required packages: 
```
pip install -r requirements.txt
```
## Dependencies
We have the following dependencies:

PyTorch 1.8.1

TorchVision 0.9.1

tensorboardX 2.5.1


## Run FlowGMM
### Cora
```
python experiments/train_flows/train_gcflow.py --gpu 0 \
--dataset CORA --pcadim 50 --alg 'flowgmm' --num_epochs 400 \
--lr 0.003 --bs 2708  --net_config "{'hidden_dim':256,'flow_layers':4,'num_transform_blocks':6, 'dropout_ratio':0.0}" \
--gauss_config "{'means_r': 1.6, 'cov_std': 1.1}" --trainer_config "{'unlab_weight':0.2}"
```
### Citeseer
```
python experiments/train_flows/train_gcflow.py --gpu 0 \
--dataset CITESEER --pcadim 100 --alg 'flowgmm' --num_epochs 400 \
--lr 0.003 --bs 3327  --net_config "{'hidden_dim':256,'flow_layers':10,'num_transform_blocks':6, 'dropout_ratio':0.1}" \
--gauss_config "{'means_r': 2.0, 'cov_std': 1.5}" --trainer_config "{'unlab_weight':0.01}"
```
### Pubmed
```
python experiments/train_flows/train_gcflow.py --gpu 0 \
--dataset PUBMED --pcadim 50 --alg 'flowgmm' --num_epochs 400 \
--lr 0.002 --bs 19717  --net_config "{'hidden_dim':256,'flow_layers':12,'num_transform_blocks':6, 'dropout_ratio':0.1}" \
--gauss_config "{'means_r': 1.6, 'cov_std': 1.1}" --trainer_config "{'unlab_weight':0.4}"
```

## Run GC-Flow
### Cora
```
python experiments/train_flows/train_gcflow.py --gpu 0 \
--dataset CORA --pcadim 50 --alg 'gcflow' --num_epochs 400 \
--lr 0.003 --bs 2708  --net_config "{'hidden_dim':256,'flow_layers':4,'num_transform_blocks':6, 'dropout_ratio':0.5}" \
--gauss_config "{'means_r': 1.6, 'cov_std': 1.1}" --trainer_config "{'unlab_weight':0.2}"
```
### Citeseer
```
python experiments/train_flows/train_gcflow.py --gpu 0 \
--dataset CITESEER --pcadim 100 --alg 'gcflow' --num_epochs 400 \
--lr 0.003 --bs 3327  --net_config "{'hidden_dim':256,'flow_layers':10,'num_transform_blocks':6, 'dropout_ratio':0.3}" \
--gauss_config "{'means_r': 1.8, 'cov_std': 1.3}" --trainer_config "{'unlab_weight':0.01}"
```

### Pubmed
```
python experiments/train_flows/train_gcflow.py --gpu 0 \
--dataset PUBMED --pcadim 50 --alg 'gcflow' --num_epochs 400 \
--lr 0.002 --bs 19717  --net_config "{'hidden_dim':256,'flow_layers':10,'num_transform_blocks':6, 'dropout_ratio':0.2}" \
--gauss_config "{'means_r': 1.6, 'cov_std': 1.1}" --trainer_config "{'unlab_weight':0.4}"
```

## Citation:
```
@INPROCEEDINGS{Wang2023,
  AUTHOR = {Tianchun Wang and Farzaneh Mirzazadeh and Xiang Zhang and Jie Chen},
  TITLE = {{GC-Flow}: A Graph-Based Flow Network for Effective Clustering},
  BOOKTITLE = {Proceedings of the Fortieth International Conference on Machine Learning},
  YEAR = {2023},
}
```

## References:
- FlowGMM: https://github.com/izmailovpavel/flowgmm
- Neural Spline Flows: https://github.com/bayesiains/nsf
