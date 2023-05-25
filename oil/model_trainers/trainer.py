import torch, dill
from torch import optim
from ..logging.lazyLogger import LazyLogger
from ..utils.utils import Eval, Named
from ..utils.mytqdm import tqdm
from ..tuning.study import guess_metric_sign
import copy, os, random
import glob
from experiments.train_flows.utils.norm_util import get_param_groups
import numpy as np
from natsort import natsorted


class Trainer(object,metaclass=Named):
    """ Base trainer
        """
    def __init__(self, model,device, dataloaders, opt_constr=optim.Adam, lr_sched = lambda e: 1, dataset=None, seed=None,
                 grad_norm_clip_value=5., log_dir=None, log_suffix='',log_args={},early_stop_metric=None):

        # Setup model, optimizer, and dataloaders
        self.device = device
        self.model = model
        self.optimizer = opt_constr(self.model.parameters())


        try: self.lr_schedulers = [lr_sched(optimizer=self.optimizer)]
        except TypeError: self.lr_schedulers = [optim.lr_scheduler.LambdaLR(self.optimizer,lr_sched)]
        self.dataloaders = dataloaders # A dictionary of dataloaders
        self.adj_mat = self.dataloaders['all'].dataset.train_adj.to(self.device)
        self.idx_all = self.dataloaders['all'].dataset.indeces_use.to(self.device)
        self.epoch = 0

        self.logger = LazyLogger(log_dir, log_suffix, **log_args)
        #self.logger.add_text('ModelSpec','model: {}'.format(model))
        self.hypers = {}
        self.ckpt = None#copy.deepcopy(self.state_dict())
        self.early_stop_metric = early_stop_metric
        self.finalacc=0
        self.bestacc=0
        self.ARI=0
        self.best_val_acc=0
        self.best_epoch=0
        self.grad_norm_clip_value = grad_norm_clip_value
        self.dataset = dataset
        self.seed = seed
        self.params_valmodel=None
        self.MI=0
        self.SScore=0
        self.macro_f1=0
        self.loss_value=[]
    
    def metrics(self,loader):
        return {}

    def loss(self, minibatch):
        raise NotImplementedError

    def train_to(self, final_epoch=100):
        assert final_epoch>=self.epoch, "trying to train less than already trained"
        self.train(final_epoch-self.epoch)

    def train(self, num_epochs=100):
        """ The main training loop"""
        self.totalep=num_epochs
        start_epoch = self.epoch
        #steps_per_epoch = len(self.dataloaders['train']); step=0
        steps_per_epoch = len(self.dataloaders['train']); step = 0
        for self.epoch in tqdm(range(start_epoch+1, start_epoch + num_epochs+1),desc='train'):

            #for i, minibatch in enumerate(self.dataloaders['train']):
            for i, minibatch in enumerate(self.dataloaders['train']):
                step = i + (self.epoch-1)*steps_per_epoch
                with self.logger as do_log:
                   if do_log: self.logStuff(step, minibatch)
                self.step(minibatch)
                [sched.step(step/steps_per_epoch) for sched in self.lr_schedulers]
        self.logStuff(step)
            

    def dynamic_train_nsf(self, num_epochs=100, inner_epoch=10):
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 250, 0)
        self.totalep = num_epochs
        start_epoch = self.epoch
        steps_per_epoch = len(self.dataloaders['train'])
        step = 0
        for self.epoch in tqdm(range(start_epoch + 1, start_epoch + num_epochs + 1), desc='train'):
            for self.inner_epoch in range(1, inner_epoch+1):
                for i, minibatch in enumerate(self.dataloaders['train']):
                    step = i + (self.epoch - 1) * steps_per_epoch
                    inner_step = self.inner_epoch -1
                    #print("inner_step {}".format(inner_step))
                    self.logStuff(step, minibatch)
                    # with self.logger as do_log:
                    #     if do_log:
                    #         self.logStuff(step, minibatch)
                    self.model.train()
                    self.lr_schedulers[0].step(step)
                    self.step(minibatch)


        self.logStuff(step)
        if step == self.totalep - 1:
            #torch.save(self.model.state_dict(), 'flowmodel_params.pt')
            print("Best acc {:.3f} in epoch {}".format(self.bestacc,self.best_epoch))
            print("Clustering MI {:.3f}".format(self.MI))
            print("Clustering ARI {:.3f}".format(self.ARI))
            print("Clustering SScore {:.3f}".format(self.SScore))




    def step(self, minibatch):
        self.optimizer.zero_grad()
        #self.scheduler.step(step)
        loss = self.loss(minibatch)
        loss.backward()
        self.optimizer.step()

        return loss

    def logStuff(self, step, minibatch=None):

        metrics = {}
        if minibatch is not None and hasattr(self,'loss'):
            try: metrics['Minibatch_Loss'] = self.loss(minibatch).cpu().data.numpy()
            except (NotImplementedError, TypeError): pass
        for loader_name,dloader in self.dataloaders.items(): # Ignore metrics on train
            if loader_name=='train' or len(dloader)==0 or loader_name[0]=='_': continue
            for metric_name, metric_value in self.metrics(dloader).items():
                metrics[loader_name+'_'+metric_name] = metric_value
        self.logger.add_scalars('metrics', metrics, step)
        schedules = {}
        for i, sched in enumerate(self.lr_schedulers):
            schedules['lr{}'.format(i)] = sched.get_lr()[0]
        self.logger.add_scalars('schedules', schedules, step)

        for name,m in self.model.named_modules():
            if hasattr(m, 'log_data'):
                m.log_data(self.logger,step,name)
        self.logger.report()
        if len(self.loss_value) <self.epoch:
            self.loss_value.append(self.loss(minibatch).cpu().data.numpy())
        # update the best checkpoint
        if self.early_stop_metric is not None:
            maximize = guess_metric_sign(self.early_stop_metric)
            sign = 2*maximize-1
            best = (sign*self.logger.scalar_frame[self.early_stop_metric].values).max()
            current = sign*self.logger.scalar_frame[self.early_stop_metric].iloc[-1]
            if current >= best: self.ckpt = copy.deepcopy(self.state_dict())
        else: self.ckpt = copy.deepcopy(self.state_dict())

    def evalAverageMetrics(self,loader, adj_mat, idx, metrics, labels=None):
        num_total, loss_totals = 0, 0
        with Eval(self.model), torch.no_grad():
            for minibatch in loader:
                try: mb_size = loader.batch_size
                except AttributeError: mb_size=1
                if labels is not None:
                    mb1_adj = (minibatch, adj_mat)
                    loss_totals += mb_size*metrics(mb1_adj, idx, labels)
                else:
                    mb1_adj = (minibatch[1], adj_mat)
                    loss_totals += mb_size * metrics(minibatch, mb1_adj, idx)
                num_total += mb_size
        if num_total==0: raise KeyError("dataloader is empty")
        return loss_totals/num_total

    def state_dict(self):
        state = {
            'outcome':self.logger.scalar_frame[-1:],
            'epoch':self.epoch,
            'model_state':self.model.state_dict(),
            'optim_state':self.optimizer.state_dict(),
            'logger_state':self.logger.state_dict(),
        }
        return state

    def load_state_dict(self,state):
        self.epoch = state['epoch']
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optim_state'])
        self.logger.load_state_dict(state['logger_state'])

    def load_checkpoint(self,path=None):
        """ Loads the checkpoint from path, if None gets the highest epoch checkpoint"""
        if not path:
            chkpts = glob.glob(os.path.join(self.logger.log_dirr,'checkpoints/c*.state'))
            path = natsorted(chkpts)[-1] # get most recent checkpoint
            print(f"loading checkpoint {path}")
        with open(path,'rb') as f:
            self.load_state_dict(dill.load(f))

    def save_checkpoint(self):
        return self.logger.save_object(self.ckpt,suffix=f'checkpoints/c{self.epoch}.state')

