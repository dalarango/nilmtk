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


sr_df = pd.read_csv('../Dataanalyse_forbruksverdier.csv', sep=';')
sr_df = sr_df.stack().str.replace(',','.').unstack()

y = sr_df['VOLUM'].iloc[0:resolution].values
y = y.astype('float64')


y = smooth(y, window_len=resolution//100, window='hanning')


signal_1 = y
signal_2 = y-1.8*y + 17*y*y
signal_3 = y-1*y + 15*y*y

agg_signal = signal_1 + signal_2 + signal_3

## Initialize



from Own_imp_SE import DSC

x = agg_signal

x_train = np.column_stack((agg_signal, signal_1))



train_set = x_train[:, 0:1]
app_data = x_train[:, 1:2]
k = x_train.shape[1] -1
T = x_train.shape[0] 
m = x_train.shape[1] -1
rp = 0.0005
epsilon = 0.001
alpha = 0.00001
steps = 100 # steps must be higher than k
n_components = x_train.shape[1] + 1
n = n_components

A = np.random.random((n,m))
B = np.random.random((T,n))


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


plt.plot(app_data, color= 'black', lw=2, linestyle='--', label='Real data', alpha=0.5)
plt.plot(x_predict, color='red', lw=2, linestyle='--', label='Reconstructed', alpha=0.5)
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.show()















comp_matrix = np.column_stack((signal_1, signal_2, signal_3))

x_train = {'main signal': comp_matrix[:,0:1], 'app_1' : comp_matrix[:,1:2], 'app_2' : comp_matrix[:,2:3]}

x_train = comp_matrix[:10000, ]
x_test  = comp_matrix[10000:, ] 


from Own_imp_SE import DSC

## Set parameters
train_set = x_train
test_set = x_test
k = x_train.shape[1]
T = x_train.shape[0]
m = x_train.shape[1]
rp = 0.0005
epsilon = 0.001
alpha = 0.00001
steps = 10 # steps must be higher than k
n_components = x_train.shape[1]

# Instanciate discriminator

dsc = DSC(train_set, alpha, epsilon, rp, steps, n_components, m, T, k)

# Pretraining
A_list, B_list = dsc.pre_training(train_set)

# Dissagregation training

# Use test set

# Check accuracy

# Dissagregation error

# Plotting



def F(self,x,B,x_train=None,A=None,rp_tep=False,rp_gl=False):
        '''
        input is lists of the elements
        output list of elements
        '''
        # 4b
        B = np.asarray(B)
        A = np.asarray(A)
        coder = SparseCoder(dictionary=B.T,
                                transform_alpha=self.rp, transform_algorithm='lasso_cd')
        comps, acts = librosa.decompose.decompose(x,transformer=coder)
        acts = self._pos_constraint(acts)

return acts



















train_set = x_train
train_sum = sum(x_train.values())
k = len(x_train.keys())
alpha = 0.00001
epsilon = 0.001
rp = 0.0005
steps = 10
n_components = 3
T,_ = x_train[list(x_train.keys())[0]].shape
m = 3

dsc = DSC(train_set,train_sum,alpha,epsilon,rp,steps,n_components,m,T,k)

A_list,B_list = dsc.pre_training(train_set.values())

x = np.concatenate(list(train_set.values())).reshape(-1, 3)

B_cat = dsc.DD(x,B_list,A_list)


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

x = np.concatenate(list(train_set.values())).reshape(-1, 1)


acts = dsc.F(x,B_cat,A=A_star)

B_cat = (B_cat-alpha*((x-B_cat.dot(acts))
                    .dot(acts.T) - (x-B_cat.dot(A_star)).dot(A_star.T)))

B_cat = dsc._pos_constraint(B_cat)
B_cat /= sum(B_cat)

# convergence check
acts_split = np.split(acts,dsc.k,axis=0)
B_split = np.split(B_cat,dsc.k,axis=1)
acc_iter = dsc.accuracy(x_train_sum,dsc.train_sum,B_list,acts_split)

x = x_train_sum
x_sum = dsc.train_sum
B = B_list
A = A_list

B_cat = np.hstack(B_list)
A_cat = np.vstack(A_list)

A_prime = dsc.F(x_sum,B_cat,A=A_cat)
A_last = np.split(A_prime,dsc.k,axis=0)
x_predict = dsc.predict(A_last,B)
acc_numerator = (map(lambda i: (np.minimum((B[i].dot(A_last[i])).sum() ,
                (sum(x[i].sum())))) ,
                range(len(B))))
acc_denominator = sum(x_predict).sum()
acc = sum(acc_numerator) / acc_denominator
acc_numerator = (map(lambda i: (np.minimum((B[i].dot(A_last[i])).sum() ,
                (sum(x[i].sum())))) ,
                range(len(B))))
acc_denominator = x_sum.values.sum()
acc_star = sum(acc_numerator) / acc_denominator























acc_iter = dsc.accuracy(x_train_sum,dsc.train_sum,B_split,A_list)
err_iter = dsc.error(x_train_sum,dsc.train_sum,B_list,acts_split)
acc_ddsc.append(acc_iter)
err_ddsc.append(err_iter)
a_ddsc.append(np.linalg.norm(acts))
b_ddsc.append(np.linalg.norm(B_cat))

change = np.linalg.norm(B_cat - B_cat_p)
t += 1
print("DD change is %f and step is %d" %(change,t))



