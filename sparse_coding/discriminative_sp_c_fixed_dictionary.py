from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from sklearn.decomposition import SparseCoder, DictionaryLearning
from sklearn.metrics import mean_squared_error


def ricker_function(resolution, center, width):
    """Discrete sub-sampled Ricker (Mexican hat) wavelet"""
    x = np.linspace(0, resolution - 1, resolution)
    x = ((2 / (np.sqrt(3 * width) * np.pi ** .25))
         * (1 - (x - center) ** 2 / width ** 2)
         * np.exp(-(x - center) ** 2 / (2 * width ** 2)))
    return x

def ricker_matrix(width, resolution, n_components):
    """Dictionary of Ricker (Mexican hat) wavelets"""
    centers = np.linspace(0, resolution - 1, n_components)
    D = np.empty((n_components, resolution))
    for i, center in enumerate(centers):
        D[i] = ricker_function(resolution, center, width)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    return D

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


def dissagregation_error(real_values, predictions):
        return mean_squared_error(real_values, predictions)


def gradient_descent(X,y, theta, learning_rate=0.01, iterations=100):
        m= len(y)
        cost_history = np.zeros(iterations)
        theta_history = np.zeros((iterations, 2))
        for it in range(iterations):
                prediction = np.dot(X,theta)
                theta = theta -(1/m)*learning_rate*(X.T.dot((prediction - y)))
                theta_history[it, :] = theta.T
                cost_history[it] = cal_cost(theta, X,y)
        return theta, cost_history, theta_history


DictionaryLearning(n_components=n_compo)

resolution = 19126
subsampling = 1800
width = 5000
n_components = resolution // subsampling

di = DictionaryLearning(n_components=n_components, fit_algorithm='cd' , transform_algorithm= 'lasso_cd',  positive_code=True, positive_dict=True)


sr_df = pd.read_csv('../Dataanalyse_forbruksverdier.csv', sep=';')
sr_df = sr_df.stack().str.replace(',','.').unstack()

y = sr_df['VOLUM'].iloc[0:resolution].values
y = y.astype('float64')


di.fit(y.reshape(-1, 1))

d = di.transform(y.reshape(-1, 1))


## Compute dictionary matrix
D_fixed = ricker_matrix(width=width, resolution=resolution,
                        n_components=n_components)

y = smooth(y, window_len=resolution//100, window='hanning')



D = D_fixed
n_nonzero = 10
alpha = None
algo = 'omp'
color_1 = 'red'
title = algo.upper()


coder_1 = SparseCoder(dictionary=d, transform_n_nonzero_coefs=n_nonzero,
                            transform_alpha=alpha, transform_algorithm=algo)


x_ = coder_1.fit_transform(y.reshape(1, -1))
density = len(np.flatnonzero(x_))
x = np.ravel(np.dot(x_, D))
squared_error = np.sum((y - x) ** 2)


plt.plot(y, color= 'black', lw=2, linestyle='--', label='Original signal', alpha=0.5)
plt.plot(x, color=color_1, lw=2,
            label='%s: %s nonzero coefs,\n%.2f error'
            % (title, density, squared_error), alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()



