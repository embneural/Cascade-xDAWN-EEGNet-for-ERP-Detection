# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:28:02 2022

@author: home
"""


#%%
import sys
import os
import torch
'''
chage src_path and root.
root is the path of the location of the datasets 
'''
# src_path = 'H:\wzh\p300_new\mycode\P300_detection'
# root = 'H:\\wzh\\p300_new\\dataset\\BCI IIb & III'

src_path = 'F:\\lab-code-backup-master\\P300_detection\\SOTA-of-P300-Detection-in-BCI-Competitions-II-III\\src'
root = 'F:\\zhendianyuanzi_linux\\dataset\\p300'
sys.path.insert(0, src_path)
#%%
from src.datasets.ERP import BCI_II_III
from src.preprocessing.time_filters import Chebyshev
from src.preprocessing.spatial_filters import xdawn
from src.preprocessing.util import normalize, expand_dim
from src.train.misc import flatten, get_CT, sample_weight, make_dataloader, Training
from src.models.EEGNet import EEGNet
from src.models.DeepConvNet import DeepConvNet
from src.train.loss import FocalLoss
#%%
order = 12
ripple = 0.1
critical_freq = 30  # unit Hz
btype= 'lowpass'
forward_backward = True
cheby_f = Chebyshev(order, ripple, critical_freq, btype, BCI_II_III.sfreq, forward_backward)

epoch_len = 0.8; detrend = 1; baseline = None; my_filter = ['my', cheby_f]; downsample = 2;
nm_type = 'z-score'; sampler = 'uniform'; batch_size = 64; device = 'cuda'
focal_gamma = 2; lr = 1e-3; weight_decay = 0

early_type = 'loss';  max_epoch = 80; patience = max_epoch;
event_dict = BCI_II_III.event_dict
save_path = None

# seed = 0
mixup = [False, 0]  #[True, 0.2], [True, 0.3], [True, 0.4]
model_type_list = ['xdawn_8_eegnet'] # 'eegnet','deepconvnet'

for model_type in model_type_list:
    if 'xdawn' in model_type:
        use_xdawn = True
        xdawn_nfilter = int( model_type.split('_')[1])
    else:
        use_xdawn =  False

    for s_id in ['B', 'A', 'C']:        
        '''
        Preprocessing
        '''
        name = {'A':'III', 'B':'III','C': 'IIb'}
        path = os.path.join(root, name[s_id])
        if s_id == 'B' or s_id == 'A':
            train_raw, train_code,  train_label = BCI_II_III.get_data_III(s_id, 'train', path)
            valid_raw, valid_code,  valid_label = BCI_II_III.get_data_III(s_id, 'test',  path)
        elif s_id == 'C':
            train_raw, train_code,  train_label = BCI_II_III.get_data_IIb('train', path)
            valid_raw, valid_code,  valid_label = BCI_II_III.get_data_IIb('test',  path)

        data = {}
        data['train'] = BCI_II_III.preprocess(epoch_len, detrend, baseline, my_filter, downsample, train_raw)
        data['valid'] = BCI_II_III.preprocess(epoch_len, detrend, baseline, my_filter, downsample, valid_raw)

        if use_xdawn:
            expand_dim( normalize(xdawn(data, xdawn_nfilter), nm_type) ) # in place
        else:
            expand_dim( normalize(data, nm_type) ) # in place
        dataloader = make_dataloader(data, sampler, batch_size)
        # recognize data
        stimulate_code = BCI_II_III.code_transfer(valid_code,       valid_label)
        recognize_data = BCI_II_III.data_transfer(data['valid'][0], valid_label)
        #%%
        '''
        Traning
        '''
        # torch.manual_seed(seed)
        weight = torch.as_tensor(sample_weight(data['train'][1]), dtype = torch.float32)
        C, T = get_CT(dataloader['train'])

        if 'eegnet' in model_type:
            model  = EEGNet(8, 2, C, T, 0.25, N = 2, filter_len = 64).to(device)
        elif 'deepconvnet' in model_type:
            model = DeepConvNet(C, T,  5, 0.5, N = 2, pool_type = 'maxpool').to(device)

        loss_fn        = [FocalLoss(weight.to(device), focal_gamma), 
                        FocalLoss(weight.to(device), 0)]
        optimizer      = torch.optim.RAdam(model.parameters(),lr = lr, weight_decay= weight_decay)  ##使用Adam优化器        
        scheduler      = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.96); 
        optimizer = [optimizer]; scheduler = [scheduler]
        vis_name = s_id + '_' + model_type + '_' + str( mixup[1] )
        tr = BCI_II_III.BCI_II_III_Training(vis_name, model, loss_fn, optimizer, scheduler, event_dict, device,
                    early_type, patience, max_epoch, save_path,
                    mixup,
                    )
        tr.early_stop(dataloader, (recognize_data, stimulate_code, valid_label))    
        tr.vis.save([vis_name])
