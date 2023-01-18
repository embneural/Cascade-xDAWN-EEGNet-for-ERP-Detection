# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:45:30 2022

@author: home
"""


import numpy as np
import torch
import torch.nn as nn



def model_params_num(model):
    num_params = sum(p.numel() for p in model.parameters())                              # Total parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters
    
    return num_params, num_trainable_params

class ConvBlock(nn.Module):
    def __init__(self,in_chs , out_chs, kernel_size):
        super(ConvBlock,self).__init__()
        
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, bias=False)
        self.bn   = nn.BatchNorm2d(out_chs)
        self.elu  = nn.ELU()
    
    def forward(self,x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        
        return x



class PoolingBlock(nn.Module):
    def __init__(self, kernel_size = (1, 2), dropout = 0.5, pool_type = 'maxpool'):
        super(PoolingBlock,self).__init__()
        
        if pool_type not in ['maxpool', 'avgpool']:
            raise ValueError('pool type error')
        
        if pool_type == 'maxpool':
        
            self.pool = nn.MaxPool2d(kernel_size)
        
        elif pool_type == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size)
            
        self.dropout = nn.Dropout(dropout)
        

    
    def forward(self,x):
        
        x = self.pool(x)
        x = self.dropout(x)
        
        return x



class DeepConvNet(nn.Module): #注意该模型没有加入maxnorm，以后再加入
    
    def __init__(self, C, T, size, dropout, N, pool_type = 'maxpool', max_norm_ratio = 1):
        super(DeepConvNet,self).__init__()
        
        self.conv = nn.Conv2d(1, 25, (1,size))   # -4
        

        self.block = nn.Sequential(
            
                ConvBlock(25, 25, (C,1)),       #
                PoolingBlock((1,2), dropout, pool_type),   # //2
                
                ConvBlock(25, 50, (1,size)),     # -4
                PoolingBlock((1,2), dropout, pool_type), # //2
                

                ConvBlock(50, 100, (1,size)),    # -4
                PoolingBlock((1,2), dropout, pool_type), # //2

                ConvBlock(100, 200, (1,size)),   # -4
                PoolingBlock((1,2), dropout, pool_type), # //2
                
            )
        
        
        a = (size-1)
        self.len_ = ((((T-a)//2 - a )//2 - a)//2-a)//2
        print(self.len_)

        self.max_norm_ratio = max_norm_ratio
        
        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.len_ * 200, N),
                
            )

    def forward(self, x):

        
        x = self.conv(x)
        x = self.block(x)
        x = self.fc(x)
        
        return x


    def max_norm(self):
        eps = 1e-8
        
        def constraint(max_val, param):            
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))


        for name, param in self.named_parameters():
            if 'bias' not in name:    # bias项不进行最大范数约束，只约束weight项
                
                if (   name == 'conv.weight' or 
                       name == 'block.0.conv.weight' or 
                       name == 'block.2.conv.weight' or 
                       name == 'block.4.conv.weight' or 
                       name == 'block.6.conv.weight'):
                    constraint(2 * self.max_norm_ratio, param)
                    
                    # print(name)
                    
                elif name == 'fc.1.weight':
                    constraint(0.5 * self.max_norm_ratio, param)
                    # print(name)


















