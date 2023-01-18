# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 17:59:35 2021

@author: wzh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    




class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, 
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)




class P300SpellerLoss(torch.nn.Module):
    def __init__(self, repetition_id, alpha, pemute_num,
                       weight=None,  gamma=0, device = 'cuda'):
        super(P300SpellerLoss, self).__init__()

        self.r = repetition_id
        self.a = alpha
        self.p = pemute_num
        self.focal_loss = FocalLoss(weight, gamma)
        self.ce_loss    = FocalLoss(gamma = 0)
        self.device = device
    def forward(self, logits, label):
        
        
        '''
        sample loss
        '''
        
        l1 = self.focal_loss(logits, label)
        self.l1 = l1
        if len( np.unique( label.cpu().numpy() ) ) != 2:
            
            return l1
        
        
        
        
        
        '''
        symbol loss
        '''
        
        Non_idx    = np.where(label.cpu().numpy() == 0 )[0]
        Target_idx = np.where(label.cpu().numpy() == 1 )[0]

        symbol_X = []
        symbol_y = []
        for j in range(self.p):
            np.random.shuffle(Non_idx)   
            np.random.shuffle(Target_idx)
            
            symbol_X.append(logits[Non_idx   [:self.r] ].mean(axis = 0, keepdims = True))
            symbol_X.append(logits[Target_idx[:self.r] ].mean(axis = 0, keepdims = True))

            symbol_y.append(0)
            symbol_y.append(1)

        
        
        X = torch.cat(symbol_X, dim = 0) .to(self.device)
        y = torch.tensor(symbol_y, dtype = torch.long).to(self.device)
        

        if torch.isnan(X).sum() != 0: # is nan

            self.X = X
            self.y = y
            self.label = label
            raise ValueError('nan value')
        

        
        l2 = self.ce_loss(X , y)
        self.l2 = l2
        
        
        
        
        return l2
        


