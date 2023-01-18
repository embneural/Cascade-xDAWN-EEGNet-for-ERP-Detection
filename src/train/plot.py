import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift


def plot_psd(y, fs):
    '''
    Parameters
    ----------
    y : array
        data.
    fs : float
        sampling frequency.

    Returns
    -------
    None.

    '''
    out = fft(y)
    power = np.abs(out)
    angle = np.angle(out)
    
    freq = np.linspace(0, fs/2, len(power)//2)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel('abs value')
    ax[0].plot(freq, power[:len(power)//2] )
        
    ax[1].set_ylabel('Phase angle')
    ax[1].plot(freq, angle[:len(angle)//2] )
    
    plt.xlabel('Frequency (Hz)')