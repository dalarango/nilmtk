from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from sklearn.decomposition import SparseCoder, DictionaryLearning
from sklearn.metrics import mean_squared_error
import librosa


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


resolution = 19126
subsampling = 1800
width = 5000
n_components = resolution // subsampling


sr_df = pd.read_csv('../Dataanalyse_forbruksverdier.csv', sep='\t')
sr_df = sr_df.stack().str.replace(',','.').unstack()

y = sr_df['VOLUM'].iloc[0:resolution].values
y = y.astype('float64')

#di = DictionaryLearning(n_components=n_components, fit_algorithm='cd' , transform_algorithm= 'lasso_cd',  positive_code=True, positive_dict=True)

#di.fit(y.reshape(1, -1))

#d = di.transform(y.reshape(1, -1))


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


coder_1 = SparseCoder(dictionary=D, transform_n_nonzero_coefs=n_nonzero,
                            transform_alpha=alpha, transform_algorithm=algo)


x_ = coder_1.fit_transform(y.reshape(1, -1))
density = len(np.flatnonzero(x_))
x = np.ravel(np.dot(x_, D))
mean_squared_error = mean_squared_error(y, x)


plt.plot(y, color= 'black', lw=2, linestyle='--', label='Original signal', alpha=0.5)
plt.plot(x, color=color_1, lw=2,
            label='%s: %s nonzero coefs,\n%.2f error'
            % (title, density, mean_squared_error), alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()





### Creating artificial subsignals

signal_1 = y

signal_2 = y - 2*y + 0.5*y*y

main_signal = signal_1 + signal_2

comp_matrix = np.vstack((main_signal, signal_1, signal_2))

comp_matrix.T

plt.plot(main_signal, color= 'black', lw=2, linestyle='--', label='Original signal', alpha=0.5)
plt.plot(signal_1, color='red', lw=2, linestyle='--', label='Appliance 1',  alpha=0.5)
plt.plot(signal_2, color='blue', lw=2, linestyle='--', label='Appliance 2',  alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()





resolution = 19126
subsampling = 1800
width = 5000
n_components = 3

D_fixed = ricker_matrix(width=width, resolution=resolution,
                        n_components=n_components)

D = D_fixed
n_nonzero = 3
alpha = None
algo = 'omp'
color_1 = 'red'
title = algo.upper()



di = DictionaryLearning(n_components=n_components, fit_algorithm='cd' , transform_algorithm= 'lasso_cd',  positive_code=True, positive_dict=True)

di.fit(comp_matrix)

d = di.transform(comp_matrix)


coder_1 = SparseCoder(dictionary=d.T, transform_n_nonzero_coefs=n_nonzero,
                            transform_alpha=alpha, transform_algorithm=algo)

comps, acts = librosa.decompose.decompose(comp_matrix,transformer=coder_1)



plt.plot(comp_matrix[0,:], color= 'black', lw=2, linestyle='--', label='Original signal', alpha=0.5)
plt.plot(acts[0,:], color='red', lw=2, linestyle='--', label='Appliance 1',  alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()


zeros_cols = np.zeros(len(y))
new_signal = 0.5*y + 12*y*y

new_comp_matrix = np.vstack((new_signal, zeros_cols, zeros_cols))

d = di.transform(new_comp_matrix)

coder_1 = SparseCoder(dictionary=d.T, transform_n_nonzero_coefs=n_nonzero,
                            transform_alpha=alpha, transform_algorithm=algo)

comps, acts = librosa.decompose.decompose(new_comp_matrix,transformer=coder_1)

plt.plot(new_comp_matrix[0,:], color= 'black', lw=2, linestyle='--', label='Original signal', alpha=0.5)
plt.plot(10000*acts[0,:], color='red', lw=2, linestyle='--', label='Appliance 1',  alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()



def D_optim(x,B,A, epsilon = 0.001, steps=10):
        '''
        Taking the parameters as x_train_use and discriminate over the
        entire region
        '''
        # 3.
        A_star = np.vstack(A)
        B_cat = np.hstack(B)
        change = 1
        t = 0
        acc_ddsc = []
        err_ddsc = []
        a_ddsc = []
        b_ddsc = []
        x_train_sum = self.train_set.values()
        while t <= self.steps and epsilon <= change:
                B_cat_p = B_cat
                # 4a
                acts = self.F(x,B_cat,A=A_star)
                # 4b
                B_cat = (B_cat-self.alpha*((x-B_cat.dot(acts))
                        .dot(acts.T) - (x-B_cat.dot(A_star)).dot(A_star.T)))
                # 4c
                # scale columns s.t. b_i^(j) = 1
                B_cat = self._pos_constraint(B_cat)
                B_cat /= sum(B_cat)

                # convergence check
                acts_split = np.split(acts,self.k,axis=0)
                B_split = np.split(B_cat,self.k,axis=1)
                acc_iter = self.accuracy(x_train_sum,self.train_sum,B,acts_split)
                acc_iter = self.accuracy(x_train_sum,self.train_sum,B_split,A)
                err_iter = self.error(x_train_sum,self.train_sum,B,acts_split)
                acc_ddsc.append(acc_iter)
                err_ddsc.append(err_iter)
                a_ddsc.append(np.linalg.norm(acts))
                b_ddsc.append(np.linalg.norm(B_cat))

                change = np.linalg.norm(B_cat - B_cat_p)
                t += 1
                print ("DD change is %f and step is %d") %(change,t)

        self.acc_ddsc = acc_ddsc
        self.err_ddsc = err_ddsc
        self.a_ddsc = a_ddsc
        self.b_ddsc = b_ddsc
        return B_cat





x_ = coder_1.fit_transform(main_signal.reshape(1, -1))
density = len(np.flatnonzero(x_))
x = np.ravel(np.dot(x_, D))
mse = mean_squared_error(y, x)

plt.plot(main_signal, color= 'black', lw=2, linestyle='--', label='Original signal', alpha=0.5)
plt.plot(x, color='red', lw=2, linestyle='--', label='Appliance 1',  alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()



plt.plot(main_signal, color= 'black', lw=2, linestyle='--', label='Original signal', alpha=0.5)
plt.plot(D[0, :], color='red', lw=2, linestyle='--', label='Comp 1',  alpha=0.5)
plt.plot(D[0, :], color='red', lw=2, linestyle='--', label='Comp 1',  alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()




###### New try######
def _pos_constraint(a):
        indices = np.where(a < 0.0)
        a[indices] = 0.0
        return a

def F(x,B,x_train=None,A=None,rp_tep=False,rp_gl=False):
        '''
        input is lists of the elements
        output list of elements
        '''
        # 4b
        B = np.asarray(B)
        A = np.asarray(A)
        coder = SparseCoder(dictionary=B.T,
                                transform_alpha=0.0005, transform_algorithm='lasso_cd')
        comps, acts = librosa.decompose.decompose(x,transformer=coder)
        acts = _pos_constraint(acts)

        return acts

F(x=y.reshape(-1,1), B=x_, A=10)

def DD(self,x,B,A):
        '''
        Taking the parameters as x_train_use and discriminate over the
        entire region
        '''
        # 3.
        A_star = np.vstack(A)
        B_cat = np.hstack(B)
        change = 1
        t = 0
        acc_ddsc = []
        err_ddsc = []
        a_ddsc = []
        b_ddsc = []
        x_train_sum = self.train_set.values()
        while t <= self.steps and self.epsilon <= change:
                B_cat_p = B_cat
                # 4a
                acts = self.F(x,B_cat,A=A_star)
                # 4b
                B_cat = (B_cat-self.alpha*((x-B_cat.dot(acts))
                        .dot(acts.T) - (x-B_cat.dot(A_star)).dot(A_star.T)))
                # 4c
                # scale columns s.t. b_i^(j) = 1
                B_cat = self._pos_constraint(B_cat)
                B_cat /= sum(B_cat)

                # convergence check
                acts_split = np.split(acts,self.k,axis=0)
                B_split = np.split(B_cat,self.k,axis=1)
                acc_iter = self.accuracy(x_train_sum,self.train_sum,B,acts_split)
                acc_iter = self.accuracy(x_train_sum,self.train_sum,B_split,A)
                err_iter = self.error(x_train_sum,self.train_sum,B,acts_split)
                acc_ddsc.append(acc_iter)
                err_ddsc.append(err_iter)
                a_ddsc.append(np.linalg.norm(acts))
                b_ddsc.append(np.linalg.norm(B_cat))

                change = np.linalg.norm(B_cat - B_cat_p)
                t += 1
                print ("DD change is %f and step is %d") %(change,t)

        self.acc_ddsc = acc_ddsc
        self.err_ddsc = err_ddsc
        self.a_ddsc = a_ddsc
        self.b_ddsc = b_ddsc
        return B_cat
