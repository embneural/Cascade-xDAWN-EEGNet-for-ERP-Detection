

import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np


class Fir():
    def __init__(self, order, critical_freq, pass_zero = 'lowpass', sample_f = None, forward_backward = True):
        
        self.b = signal.firwin(order, critical_freq, window = 'hamming', pass_zero= pass_zero, fs = sample_f) #计算出滤波器的系数
        self.a = 1
        self.forward_backward = forward_backward
        self.sample_f = sample_f
    
    def apply(self, X):
        if self.forward_backward:
            return signal.filtfilt(self.b, self.a, X) # forward and backward ， 这个是0相位
        else:
            return signal.lfilter(self.b, self.a, X) # 这个也是应用滤波参数，只进行forward，只是普通模式

    def check(self):
        
        fig, ax1 = plt.subplots()
        ax1.set_title('Digital filter frequency response')
        w, h = signal.freqz(self.b, self.a)
        w = w/np.pi * self.sample_f/2.0
        
        ax1.plot(w, 20 * np.log10(abs(h)), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
        ax1.set_xlabel('Frequency [Hz]')
        
        
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')
        plt.show()
        






class comb():
    def __init__(self, f_removed, Q = 30,  sample_f = None, forward_backward = True):
        

        self.b, self.a = signal.iircomb(f_removed, Q,  ftype='notch', fs = sample_f)
        self.forward_backward = forward_backward
        self.fs = sample_f
        
    def apply(self, X):
        if self.forward_backward:
            return signal.filtfilt(self.b, self.a, X) # forward and backward ， 这个是0相位
        else:
            return signal.lfilter(self.b, self.a, X) # 这个也是应用滤波参数，只进行forward，只是普通模式

    def check(self):
        

        freq, h = signal.freqz(self.b, self.a, fs=self.fs)
        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='blue')
        ax[0].set_xlim([0, 100])
        ax[0].set_ylim([-25, 10])
        ax[0].grid()
        ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
        ax[1].set_ylabel("Angle (degrees)", color='green')
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_xlim([0, 100])
        ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax[1].set_ylim([-90, 90])
        ax[1].grid()
        plt.show()    








class Chebyshev():
    def __init__(self, order, ripple, critical_freq, btype, sampling_freq, forward_backward = True):
    
        
        self.b, self.a = signal.cheby1(N = order, rp = ripple, Wn = critical_freq, btype= btype, output='ba', fs = sampling_freq)
        self.forward_backward = forward_backward
        self.sample_f = sampling_freq
        self.critical_freq = critical_freq
        self.ripple = ripple
        
    def apply(self, X):
        if self.forward_backward:
            return signal.filtfilt(self.b, self.a, X) # forward and backward ， 这个是0相位
            
        else:
            return signal.lfilter(self.b, self.a, X) # 这个也是应用滤波参数，只进行forward，只是普通模式
        

    def check(self):


        fig, ax1 = plt.subplots()
        ax1.set_title('Digital filter frequency response')
        w, h = signal.freqz(self.b, self.a)
        w = w/np.pi * self.sample_f/2.0
        
        ax1.plot(w, 20 * np.log10(abs(h)), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
        ax1.set_xlabel('Frequency [Hz]')
        
        
        
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')
        plt.show()       




class butter():
    def __init__(self, order = 8, bands = [1, 90], sample_f = None, forward_backward = True):
        
                
        
        self.forward_backward = forward_backward
        self.fs = sample_f
        self.b, self.a = signal.butter(order, bands, btype='bandpass', fs = self.fs)
        
        
    def apply(self, X):
        if self.forward_backward:
            return signal.filtfilt(self.b, self.a, X) # forward and backward ， 这个是0相位
        else:
            return signal.lfilter(self.b, self.a, X) # 这个也是应用滤波参数，只进行forward，只是普通模式

    def check(self):
        

        freq, h = signal.freqz(self.b, self.a, fs=self.fs)
        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='blue')
        ax[0].set_xlim([0, 100])
        ax[0].set_ylim([-25, 10])
        ax[0].grid()
        ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
        ax[1].set_ylabel("Angle (degrees)", color='green')
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_xlim([0, 100])
        ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax[1].set_ylim([-90, 90])
        ax[1].grid()
        plt.show()    





class Notch():
    def __init__(self,f0, Quality, sampling_freq, forward_backward = True):
        
        # Design notch filter  # # Frequency to be removed from signal (Hz)
        self.b, self.a = signal.iirnotch(f0, Quality, sampling_freq)  # Quality factor: Q = f0/bandwidth
        self.forward_backward = forward_backward
        self.sample_f = sampling_freq
    def apply(self, X):
        if self.forward_backward:
            return signal.filtfilt(self.b, self.a, X) # forward and backward ， 这个是0相位
            
        else:
            return signal.lfilter(self.b, self.a, X) # 这个也是应用滤波参数，只进行forward，只是普通模式


    def check(self):

        
        # Frequency response
        freq, h = signal.freqz(self.b, self.a, fs=self.sample_f)
        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        epsilon = 1e-8
        ax[0].plot(freq, 20*np.log10(abs(h)+epsilon ), color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='blue')
        ax[0].set_xlim([0, 100])
        ax[0].set_ylim([-25, 10])
        ax[0].grid()
        ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
        ax[1].set_ylabel("Angle (degrees)", color='green')
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_xlim([0, 100])
        ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax[1].set_ylim([-90, 90])
        ax[1].grid()
        plt.show()