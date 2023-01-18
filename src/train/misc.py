
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:21:15 2021

@author: wzh
"""

from collections import Counter
import torch
from torch import nn
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader,TensorDataset
from copy import deepcopy
from torch.autograd import Variable # 获取变量
from collections import Counter
from sklearn.metrics import multilabel_confusion_matrix
import itertools
import visdom
from torch.optim.swa_utils    import AveragedModel, SWALR
import torch.nn.functional as F
from sklearn.metrics import classification_report



def flatten(l: list):
	return [item for sublist in l for item in sublist]

def get_CT(dataloader):
    """get the channels and time samples of train data.
    Parameters
    ----------
    dataloader: DataLoader
    
    Returns
    -------
    C: int
    	channels
    T: int
    	time samples
    """  

    for batch, (X, y) in enumerate(dataloader):
                
        C = X.shape[2]
        T = X.shape[3]    
        break
    
    return C, T


def sample_weight(train_y: np.array):
    
    """get train sample weight.
    Parameters
    ----------
    train_y: 1D array
        train_y.shape = (n_samples, )
    Returns
    -------
      weight: np.array
        train sample weights
    """  

    keys = []
    for key in Counter(train_y).keys():
        keys.append(key)
    keys.sort(reverse = False)  # 升序
    class_count = []
    for key in keys:
        class_count.append(Counter(train_y)[key])
    class_count = np.array(class_count)
    weight = np.max(class_count)/class_count    
    return weight



def make_dataloader(data: dict, sampler:str, batch_size:int):

    """ create dataloader
    Parameters
    ----------
    data: dict : {'train': [train_x, train_y], 'valid': [valid_x, valid_y]  }
        train_x.shape = (n_samples, 1, channels, times), 
        train_y.shape = (n_samples, )
    
    sampler: str : ['uniform', 'reverse']
        defalut is 'uniform'

    batch_size: int
        defalut is 64

    Returns
    -------
    dataloader: dict : {'train': [trainloader], 'valid':[validloader]}
    """ 
    
    if sampler not in['uniform', 'reverse']:
        raise ValueError("wrong sampler")

    if sampler == 'reverse':


        weight = sample_weight(data['train'][1])
        proba = weight/np.sum(weight)
        samples_proba = torch.from_numpy(np.array( [ proba[i] for i in  data['train'][1] ] ))
        from torch.utils.data import WeightedRandomSampler
        weight_sampler = WeightedRandomSampler(samples_proba, len(samples_proba))

    if sampler == 'reverse':
        method  = weight_sampler
        shuffle = False
        
    elif sampler == 'uniform':
        method  = None
        shuffle = True
    

    dataloader = {}
    for key in data.keys():
        
        if key == 'train':
            temp =     TensorDataset(torch.tensor(data[key][0], dtype = torch.float32), torch.tensor(data[key][1], dtype = torch.long))
            dataloader[key] =  DataLoader(dataset=temp ,batch_size=batch_size, sampler = method, shuffle=shuffle )
        
        else:
            temp = TensorDataset(torch.tensor(data[key][0], dtype = torch.float32), torch.tensor(data[key][1], dtype = torch.long))
            dataloader[key] =  DataLoader(dataset=temp ,batch_size=batch_size, shuffle=False )
        
    return dataloader

    
def confusion_matraix_to_standard(confusion_matraix):
    
    temp = confusion_matraix[:, 0, 0].copy() #要想让confusion_matraix改动不影响到temp，就必须使用.copy()
    confusion_matraix[:, 0, 0] = confusion_matraix[:, 1, 1]
    confusion_matraix[:, 1, 1] = temp
    
    temp = confusion_matraix[:, 0, 1].copy() #要想让confusion_matraix改动不影响到temp，就必须使用.copy()
    confusion_matraix[:, 0, 1] = confusion_matraix[:, 1, 0]
    confusion_matraix[:, 1, 0] = temp
    
    return confusion_matraix

    

class Training():
    def __init__(self,  vis_name, model, loss_fn:list, optimizer:list, scheduler:list, event_dict:dict, device = 'cuda',
                        early_type = 'loss', patience:int = 20, max_epoch:int = 80, save_path = None,
                        mixup:list = [False, 0],
                        swa_start = np.inf, swa_lr = 1e-3,
                        other_batch_source = False, gradient_clip = False,
                        warmup = [False, 0],
                        ):
        
        """ 
        Parameters
        ----------
        early_type: str : ['loss', 'acc', 'recall']
        function  : class | None 
            addtion function, default is None
            
            
        using_decoder: bool
            for capsule network
        """
        
        self.warmup_val = warmup[0]; self.warmup_dur = warmup[1]
        
        self.vis = visdom.Visdom(env= vis_name)
        self.vis.delete_env(vis_name)
        self.vis = visdom.Visdom(env= vis_name)
                
        
        self.model = model
        self.train_loss_fn = loss_fn[0]
        self.valid_loss_fn = loss_fn[1]
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.event_dict    = event_dict
        self.device        = device
        '''
        Tricks
        '''
        self.mix_up = mixup[0]
        self.alpha  = mixup[1]
        
        
        self.swa_start     = swa_start 
        if self.swa_start != np.inf:
            self.swa_model     = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr= swa_lr)
        
        '''
        reach patience paramters
        '''
        if early_type not in ['loss', 'acc', 'recall']:
            raise ValueError("early stop type must be 'loss', 'acc','recall' ")
        self.trigger_times = 0
        self.min_loss = np.inf
        self.early_type = early_type
        self.patience  = patience
        self.max_epoch = max_epoch
        self.save_path = save_path
        
        self.other_batch_source = other_batch_source        
        self.gradient_clip = gradient_clip
        
    def mixup_data(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def early_stop(self,data_loader:dict):

        """ 
        using early stop to train
        """
        for self.epoch in range(self.max_epoch):

            '''
            train
            '''
            train_loss, train_y, train_pred_y = self.train(self.model, data_loader['train'])
            '''
            scheduler
            '''
            if self.epoch > self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                # Update bn statistics for the swa_model at the end
                '''
                update bn every time when the swa model' weihts update, maybe this way is good
                '''
                self.update_bn(data_loader['train'], self.swa_model, device = self.device)
                self.got_model = self.swa_model
                
            else:
                
                if self.epoch < self.warmup_dur:
                    if self.warmup_val:
                        for opt in self.optimizer:
                            opt.param_groups[0]['lr'] = self.warmup_val
                        
                        if self.epoch == self.warmup_dur - 1:

                            for i, opt in enumerate(self.optimizer):
                                opt.param_groups[0]['lr'] = opt.param_groups[0]['initial_lr']                      
                else:
                    for sch in self.scheduler:
                        sch.step()
                self.got_model = self.model



            '''
            valid
            '''
            self.valid_loss, valid_y, valid_pred_y  = self.evaluate(self.got_model, data_loader['valid'])
            '''
            compute performance
            '''
            train_recall = self.compute_recall(self.event_dict, train_y, train_pred_y)
            train_acc    = (train_y == train_pred_y).sum()/ len(train_y)
            self.valid_recall = self.compute_recall(self.event_dict, valid_y, valid_pred_y)
            self.valid_acc    = (valid_y == valid_pred_y).sum()/ len(valid_y)
            '''
            patience
            '''
            if self.reach_patience(self.early_type):
                break
            '''
            save
            '''
            if self.save_path != None:
                self.pth_save(self.save_path)
            '''
            dynamic plot
            '''
            title = 'loss'
            self.vis.line(
                     X = np.column_stack((
                             self.epoch,
                             self.epoch,
                        )),
                     Y = np.column_stack((
                             train_loss,
                             self.valid_loss,
                         )),
                     opts = { 
                            'title'   :  title,
                            'legend'  :  ['train', 'valid'],
                             'dash': np.array(['solid', 'solid']),
                             },
                     win = title,
                     update='append',
                 )

            title = 'Acc'
            self.vis.line(
                    X = np.column_stack((
                            self.epoch,
                            self.epoch,
                        )),
                    Y = np.column_stack((
                            np.array(train_acc),
                            np.array(self.valid_acc),
                        )),
                    
                    opts = { 
                            'title'   :  title,
                            'legend'  :  ['train', 'valid'],
                            },
                    
                    win = title,
                    update='append',
                )
            
            title = 'marcro recall'
            self.vis.line(
                     X = np.column_stack((
                             self.epoch,
                             self.epoch,
                        )),
                     Y = np.column_stack((
                             np.array(train_recall).mean(),
                             np.array(self.valid_recall).mean(),
                             
                         )),
                     
                     opts = { 
                            'title'   :  title,
                            'legend'  :  ['train', 'valid'],
                              # 'dash': np.array(['solid', 'dash']),
                             },
                     
                     win = title,
                     update='append',
                 )

            title = 'train recall'
            self.vis.line(
                    X = np.array([self.epoch] * len(self.event_dict.keys() ) ).T [None, :]  , 
                    Y = np.array(train_recall).T [None, :],  
                    opts = { 
                            'title'   :  title,
                            'legend'  :  [key for key in self.event_dict.keys()],
                            },
                    win = title,
                    update='append',
                )
            
            title = 'valid recall'
            self.vis.line(
                    X = np.array([self.epoch] * len(self.event_dict.keys() ) ).T [None, :]  , 
                    Y = np.array(self.valid_recall).T [None, :],  
                    opts = { 
                            'title'   :  title,
                            'legend'  :  [key for key in self.event_dict.keys()],
                            },
                    win = title,
                    update='append',
                )


    
    def train(self, model, dataloader):

        model.train() # model的训练模式，保证batchnorm和dropout的参数是进行训练计算的
        loss_sum = 0; num_batches = 0        
        y_true = [];  y_pred = []

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            '''
            mix up
            '''
            if self.mix_up:
                inputs, targets_a, targets_b, lam = self.mixup_data(X, y)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                              targets_a, targets_b))

                pred = model(X)  
                _loss = self.mixup_criterion(self.train_loss_fn, model(inputs), targets_a, targets_b, lam)
                
            else:
                

                pred = model(X)     
                _loss = self.train_loss_fn(pred, y)

        
            
            for opt in self.optimizer:
                opt.zero_grad()
            _loss.backward() # flooding
            
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

            
            for opt in self.optimizer:
                opt.step()
            if hasattr(model, 'max_norm'):
                model.max_norm()

            loss_sum += _loss.item()
            num_batches += 1

            y_true.append( y.cpu().numpy())
            y_pred.append( pred.argmax(1).cpu().numpy())

        y_true = np.array(list(itertools.chain.from_iterable(y_true))) #把list展平
        y_pred = np.array(list(itertools.chain.from_iterable(y_pred))) #把list展平

        return loss_sum/num_batches, y_true, y_pred


    def evaluate(self, model, dataloader):

        model.eval()
        loss_sum = 0; num_batches = 0            
        y_pred = [];  y_true = []        
        
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)


                pred = model(X)     
                _loss = self.valid_loss_fn(pred, y)


                loss_sum += _loss.item()
                num_batches += 1

                y_true.append( y.cpu().numpy())
                y_pred.append( pred.argmax(1).cpu().numpy())
            
        y_true = np.array( list(itertools.chain.from_iterable(y_true)) ) #把list展平
        y_pred = np.array( list(itertools.chain.from_iterable(y_pred)) ) #把list展平

        return loss_sum/num_batches, y_true, y_pred


    
    def get_report(self, y_true, y_pred):
        self.report = classification_report (y_true, y_pred, output_dict=True)


        



    def compute_recall(self, event_dict, y_true, y_pred):
        
        """ compute recall
        Parameters
        ----------
        event_dict: dict
            mne event dict
        y_true: list
            label 
        y_pred: list
            pred label

        Returns
        -------
        recall: list
            
        """ 
        labels = []
        for key in event_dict.keys():
            labels.append(event_dict[key])

        # to int, and make the smallest label to zero
        labels = np.array(labels, dtype = int)
        labels = labels - np.min(labels)
                
        confusion_matraix = multilabel_confusion_matrix(y_true, y_pred, 
                                                        labels= labels)
        confusion_matraix = confusion_matraix_to_standard(confusion_matraix)

        recall = []
        for i in range(len(event_dict.keys()) ):
            TP = confusion_matraix[i, 0, 0]
            FN = confusion_matraix[i, 0, 1]
            recall.append( TP/(TP+FN) )

        return recall

    def update_bn(self, loader, model, device=None):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.
        """
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
    
        if not momenta:
            return
    
        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0
    
        for input in loader:
            if isinstance(input, (list, tuple)):
                input = input[0]
            if device is not None:
                input = input.to(device)
            
            if self.other_batch_source:
                input = input[0]
            model(input)

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)


    def reach_patience(self, type):
        '''
        patience
        '''
        if type == 'loss':
            value = self.valid_loss
        elif type == 'acc':
            value = - self.valid_acc
        elif type == 'recall':
            value = - np.mean(self.valid_recall)

        if value > self.min_loss:
            self.trigger_times += 1
            # print('trigger times:', trigger_times)
            if self.trigger_times >= self.patience:
                print('Reach patience, Early stopping!')
                return True
        else:
            # print('trigger times: 0')
            self.trigger_times  = 0
            self.min_loss = value
            return False


    def pth_save(self, save_path):
        """ 
        Parameters
        ----------
        save_path: dict : {'loss': path1, 'acc': path2, 'recall': path3}
        """

        for key in save_path.keys():
            if key not in ['loss', 'acc', 'recall']:
                raise ValueError("key must be 'loss', 'acc','recall' ")
            if key == 'loss':
                value = self.valid_loss
            elif key == 'acc':
                value = -self.valid_acc
            elif key == 'recall':
                value = - np.mean(self.valid_recall)

        if value <= self.min_loss:
            self.got_model.eval()
            torch.save({
                        'model_state_dict':     self.got_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch':        self.epoch,
                        'valid loss':   self.valid_loss,
                        'valid acc'   : self.valid_acc,
                        'valid recall': self.valid_recall,
                        }, save_path[key])
