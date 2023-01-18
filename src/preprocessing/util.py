
import numpy as np

def normalize(data:dict, method:str):

    """normalization (in place).
    Parameters
    ----------
    data: dict : {'train': [train_x, train_y], 'valid': [valid_x, valid_y]  }
        train_x.shape = (n_samples, channels, times), 
        train_y.shape = (n_samples, )
    method:str:  ['scaling_to_range', 'clipping', 'log_scaling', 'z-score', 'sigmoid']
        reference: https://developers.google.com/machine-learning/data-prep/transform/normalization
    Returns
    -------
      data: normalized data
    """  

    if method not in['scaling_to_range', 'clipping', 'log_scaling', 'z-score', 'sigmoid']:
        raise ValueError("wrong normalization methord")
    
    if method == 'scaling_to_range':
        
        '''
        When the feature is more-or-less uniformly distributed across a fixed range.
        '''
        for key in data.keys():
            xmin = np.min(data[key][0],axis= -1,keepdims=True)
            xmax = np.max(data[key][0],axis= -1,keepdims=True)
            data[key][0] = (data[key][0]-xmin)/(xmax-xmin)
        
                        
        
    elif method == 'clipping':
        
        '''
        When the feature contains some extreme outliers.
        '''
        raise ValueError("not implemented yet")
      
      
    elif method == 'log_scaling':
        
        '''
        When the feature conforms to the power law.
        '''
        
        raise ValueError("not implemented yet")
        
    elif method == 'z-score':
    
        '''
        When the feature distribution does not contain extreme outliers.
        '''
        for key in data.keys():
            mean = np.mean(data[key][0],axis= -1,keepdims=True)
            std  = np.std (data[key][0],axis= -1,keepdims=True)
            data[key][0] = (data[key][0]-mean)/(std + 1e-8)
    
    
    elif method == 'sigmoid':
        
        for key in data.keys():
            data[key][0] = 1/(1 + np.exp(-data[key][0]))
  
    
    
  
    return data





def norm(X:np.array, method:str):
    
    '''
    X : np.array
        (n_samples, n_channels, n_times)
    '''
    

    if method not in['scaling_to_range', 'z-score', 'sigmoid']:
        raise ValueError("wrong normalization methord")
        
    if method == 'scaling_to_range':
        xmin = np.min(X,axis= -1,keepdims=True)
        xmax = np.max(X,axis= -1,keepdims=True)
        X =  (X-xmin)/(xmax-xmin)
        
    elif method == 'z-score':
        '''
        When the feature distribution does not contain extreme outliers.
        '''
        mean = np.mean( X, axis= -1, keepdims=True)
        std  = np.std ( X, axis= -1, keepdims=True)
        X = (X - mean)/(std + 1e-8)
        
    elif method == 'sigmoid':
        X = 1/(1 + np.exp(-X))

    return X



        
       
    
    
def expand_dim(data:dict):

    """expand_dim (in place).
    Parameters
    ----------
    data: dict : {'train': [train_x, train_y], 'valid': [valid_x, valid_y]  }
        train_x.shape = (n_samples, channels, times), 
        train_y.shape = (n_samples, )
    Returns
    -------
      data: dict : {'train': [train_x, train_y], 'valid': [valid_x, valid_y]  }
        train_x.shape = (n_samples, 1, channels, times), 
        train_y.shape = (n_samples, )
    """  

    for key in data.keys():
        data[key][0] = np.expand_dims(data[key][0], axis = 1)
    
    return data

