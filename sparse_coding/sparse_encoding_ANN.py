from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from sklearn.decomposition import SparseCoder, DictionaryLearning
from sklearn.metrics import mean_squared_error
import librosa
import time
from Own_imp_SE import DSC


def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]



if __name__ == "__main__":
    
        resolution = 19126
        subsampling = 1800
        width = 5000


        sr_df = pd.read_csv('../Dataanalyse_forbruksverdier.csv', sep=';')
        sr_df = sr_df.stack().str.replace(',','.').unstack()

        y1 = sr_df['VOLUM1'].iloc[0:resolution].values
        y1 = y1.astype('float64')

        y2 = sr_df['VOLUM2'].iloc[0:resolution].values
        y2 = y2.astype('float64')

        y3 = sr_df['VOLUM3'].iloc[0:resolution].values
        y3 = y3.astype('float64')


        y1 = smooth(y1, window_len=resolution//100, window='hanning')
        y2 = smooth(y2, window_len=resolution//100, window='hanning')
        y3 = smooth(y3, window_len=resolution//100, window='hanning')


        signal_1 = y1
        signal_2 = y2
        signal_3 = y3


        agg_signal = signal_1 + signal_2 + signal_3

        ## Initialize
        x = agg_signal

        x_train = np.column_stack((agg_signal, signal_3))

        np.any(np.isnan(x_train))
        np.all(np.isfinite(x_train))


        train_set = x_train[:, 0:1]
        app_data = x_train[:, 1:2]
        k = x_train.shape[1] -1
        T = x_train.shape[0] 
        m = x_train.shape[1] -1
        rp = 0.0005
        epsilon = 0.01
        alpha = 0.00001
        steps = 100 # steps must be higher than k
        n_components = x_train.shape[1] + 1
        n = n_components

        #np.random.seed(101)
        #A = np.random.random(size=(n,m))
        #B = np.random.random(size=(T,n))

        A = np.load("A.npy")
        B = np.load("B.npy")

        dsc = DSC(train_set, alpha, epsilon, rp, steps, n_components, m, T, k)


        ### Initial dict

        #D_fixed = ricker_matrix(width=width, resolution=resolution,
        #                        n_components=n_components)
        #D_fixed = DSC._pos_constraint(D_fixed)
        #coder = SparseCoder(dictionary=D_fixed.T,
        #                            transform_alpha=rp, transform_algorithm='lasso_cd')
        #x_ = coder.transform(train_set)
        #A_list, B_list = dsc.pre_training(no_appliances=1)


        B_cat, theta = dsc.DD(train_set, B, A, app_data)


        ###############################################################
        #acts = dsc.F(train_set, np.hstack(B_list), A=np.vstack(A_list))
        ###############################################################

        A_prime = dsc.F(train_set, B_cat, A=np.vstack(A))

        x_predict = theta * B_cat.dot(A_prime)


        plt.plot(app_data, color= 'black', lw=2, linestyle='--', label='Real data', alpha=0.6)
        plt.plot(x_predict, color='red', lw=2, linestyle='--', label='Reconstructed', alpha=0.5)
        plt.axis('tight')
        plt.legend(shadow=False, loc='best')
        plt.show()




