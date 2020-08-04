from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from sklearn.decomposition import SparseCoder, DictionaryLearning


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


## Parameters

resolution = 19126
subsampling = 800
width = 5000
n_components = resolution // subsampling

sr_df = pd.read_csv('Dataanalyse_forbruksverdier.csv', sep=';')
sr_df = sr_df.stack().str.replace(',','.').unstack()

## Compute dictionary matrix
D_fixed = ricker_matrix(width=width, resolution=resolution,
                        n_components=n_components)

D_multi = np.r_[tuple(ricker_matrix(width=w, resolution=resolution,
                      n_components=n_components // 5)
                for w in (10, 50, 100, 500, 1000, 5000, 12000, 19000))]

### Original signal

#y = np.linspace(0, resolution - 1, resolution)
#first_quarter = y < resolution / 4
#y[first_quarter] = 3.
#y[np.logical_not(first_quarter)] = -1.

### Sin wave signal

#y = np.sin(np.arange(0, 10, 0.009765625))
#y = y * np.random.normal(loc=1.0, scale=0.10, size=resolution)

### Stig Rune's signal

y = sr_df['VOLUM'].iloc[0:resolution].values
y = y.astype('float64')

#xx = np.linspace(1,resolution, resolution)
#itp = interp1d(xx, y, kind='linear')
#window_size, poly_order = 51, 3
#y = savgol_filter(itp(xx), window_size, poly_order)

y = smooth(y, window_len=resolution//100, window='hanning')

# List the different sparse coding methods in the following format:
# (title, transform_algorithm, transform_alpha,
#  transform_n_nozero_coefs, color)
estimators = [('OMP', 'omp', None, 15, 'navy'),
              ('Lasso', 'lasso_lars', 2, None, 'turquoise'), ]
lw = 2

# Avoid FutureWarning about default value change when numpy >= 1.14
lstsq_rcond = None if LooseVersion(np.__version__) >= '1.14' else -1


D = D_fixed
n_nonzero = 15
alpha = None
algo = 'omp'
color_1 = 'red'
title = algo.upper()

coder_1 = SparseCoder(dictionary=D, transform_n_nonzero_coefs=n_nonzero,
                            transform_alpha=alpha, transform_algorithm=algo)

x_ = coder_1.transform(y.reshape(1, -1))
density = len(np.flatnonzero(x_))
x = np.ravel(np.dot(x_, D))
squared_error = np.sum((y - x) ** 2)


D = D_multi
n_nonzero = 8
alpha = None
algo = 'omp'
color_2 = 'green'
title = algo.upper()

coder_2 = SparseCoder(dictionary=D, transform_n_nonzero_coefs=n_nonzero,
                            transform_alpha=alpha, transform_algorithm=algo)

z_ = coder_2.transform(y.reshape(1, -1))
density = len(np.flatnonzero(z_))
z = np.ravel(np.dot(z_, D))
squared_error = np.sum((y - z) ** 2)




plt.plot(y, color= 'black', lw=lw, linestyle='--', label='Original signal', alpha=0.5)
plt.plot(x, color=color_1, lw=lw,
            label='%s: %s nonzero coefs,\n%.2f error'
            % (title, density, squared_error), alpha=0.5)
plt.plot(z, color=color_2, lw=lw,
            label='%s: %s nonzero coefs,\n%.2f error'
            % (title, density, squared_error), alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()
