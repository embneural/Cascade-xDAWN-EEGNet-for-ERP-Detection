



import numpy as np
import torch
import torch.nn as nn

def model_params_num(model):
    num_params = sum(p.numel() for p in model.parameters())                              # Total parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters
    
    return num_params, num_trainable_params


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True):
        super(SeparableConv2d,self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=False)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1, bias = bias)
    
    def forward(self,x):
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class EEGNet(nn.Module): #注意该模型没有加入maxnorm，以后再加入
    
    def __init__(self,  F1, D, C, T, dropout, N, filter_len = 64):
        super(EEGNet,self).__init__()
        
                
        
        f_s = [(1,filter_len), (C,1), (1,4), (1,16),(1,8)]
        

        self.f_s = f_s
        l = int( np.floor((f_s[0][1]-1)/2) )
        r = int( np.ceil((f_s[0][1]-1)/2)  )
        self.pad1 = nn.ConstantPad1d((l,r) , 0)
        self.conv1 = nn.Conv2d( 1, F1, f_s[0], bias = False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        '''
        *****************************************************
        '''
        self.depthwiseconv = nn.Conv2d(F1, D*F1, f_s[1], groups = F1, bias = False)
        # 没有加入最大范数正则化
        self.batchnorm2 = nn.BatchNorm2d(D*F1)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(f_s[2], stride = (1,4))
        self.dropout1 = nn.Dropout(dropout)
        
        '''
        *****************************************************
        '''
        
        l2 = int( np.floor((f_s[3][1]-1)/2) )
        r2 = int( np.ceil((f_s[3][1]-1)/2) )
        
        
        F2 = D*F1
        self.pad2 = nn.ConstantPad1d((l2,r2) , 0)
        self.separableconv = SeparableConv2d(D*F1, F2, f_s[3],bias = False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d(f_s[4], stride = (1,8))


        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(F2*(T//32), N),
                #没有加入max norm
            )


        self.softmax = nn.Softmax(dim = -1)
        self.logsoftmax = nn.LogSoftmax(dim = -1)
        
    def forward(self,x):

        
        x = self.pad1(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.batchnorm1(x)
                
        '''
        *****************************************************
        '''
    
        x = self.depthwiseconv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        '''
        *****************************************************
        '''
        
        x = self.pad2(x)
        x = self.separableconv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)        
        x = self.avgpool2(x)
        
        '''
        *****************************************************
        '''
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
                if ( name == 'depthwiseconv.weight'):
                    # constraint(1, param)
                    constraint(1, param)
                    # print(name)
                elif (name == 'fc.1.weight'):
                    constraint(0.25, param)
                    # print(name)











import torch.nn.functional as F
if __name__ == "__main__":

    model = EEGNet(8, 2, 64, 250, 0.2, 2)
    print(model.f_s)
    print('total parameters:', model_params_num(model)[0])
    
    
    model.conv1.weight.data.clone().detach().cpu().numpy()
    
    
    pass
    
    
    
    
    
    
    