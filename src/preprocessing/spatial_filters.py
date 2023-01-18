
import pyriemann

def xdawn(data:dict, nfilter: int):
    
    """apply xdawn spatial filter (in place). Only for ERP data.
    Parameters
    ----------
    data: dict : {'train': [train_x, train_y], 'valid': [valid_x, valid_y]  }
        train_x.shape = (n_samples, channels, times), 
        train_y.shape = (n_samples, )
    Returns
    -------
      data: dict : {'train': [train_x, train_y], 'valid': [valid_x, valid_y]  }
        train_x.shape = (n_samples, nclass*nfilter, times), 
        train_y.shape = (n_samples, )
    """ 

    xd = pyriemann.spatialfilters.Xdawn(nfilter)

    for key in data.keys():
        if key == 'train':
            xd.fit(data[key][0], data[key][1])
            data[key][0] = xd.transform(data[key][0])
        else:
            data[key][0] = xd.transform(data[key][0])

    return data


