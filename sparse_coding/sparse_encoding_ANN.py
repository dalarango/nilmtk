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

resolution = 19126
subsampling = 1800
width = 5000



sr_df = pd.read_csv('../Dataanalyse_forbruksverdier.csv', sep='\t')
sr_df = sr_df.stack().str.replace(',','.').unstack()

y = sr_df['VOLUM'].iloc[0:resolution].values
y = y.astype('float64')


y = smooth(y, window_len=resolution//100, window='hanning')


signal_1 = y
signal_2 = y-0.5*y + 2*y*y
main_signal = signal_1 + signal_2

comp_matrix = np.column_stack((main_signal, signal_1, signal_2))

x_train = {'main signal': comp_matrix[:,0:1], 'app_1' : comp_matrix[:,1:2], 'app_2' : comp_matrix[:,2:3]}

train_set = x_train
train_sum = sum(x_train.values())
k = len(x_train.keys())
alpha = 0.00001
epsilon = 0.001
rp = 0.0005
steps = 10
n_components = 3
T,m = x_train[list(x_train.keys())[0]].shape


dsc = DSC(train_set,train_sum,alpha,epsilon,rp,steps,n_components,m,T,k)

A_list,B_list = dsc.pre_training(train_set.values())


#B_cat = dsc.DD(np.concatenate(list(train_set.values())).reshape(len(comp_matrix), 3),B_list,A_list)


A_star = np.vstack(A_list)
B_cat = np.hstack(B_list)
change = 1
t = 0
acc_ddsc = []
err_ddsc = []
a_ddsc = []
b_ddsc = []
x_train_sum = train_set.values()

B_cat_p = B_cat

x = np.concatenate(list(train_set.values())).reshape(len(comp_matrix), -1)


acts = dsc.F(x,B_cat,A=A_star)

zeroes = np.zeros((len(A_star), 1))


A_star = np.concatenate((A_star, zeroes, zeroes), axis=1)

B_cat = (B_cat-alpha*((x-B_cat.dot(acts))
                    .dot(acts.T) - (x-B_cat.dot(A_star)).dot(A_star.T)))

B_cat = (B_cat-alpha*((x-B_cat.dot(acts))
                    .dot(acts.T) - (x-B_cat.dot(A_star)).dot(A_star.T)))

