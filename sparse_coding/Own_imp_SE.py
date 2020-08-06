import numpy as np
import pandas as pd
import time
import librosa
import pickle
from sklearn.decomposition import SparseCoder,DictionaryLearning
from sklearn import cluster
import matplotlib.pyplot as plt


class DSC():

    def __init__(self, train_set, gradient_step_size, epsilon, regularization_parameter, steps, n_components, m, T, k):
        
        self.train_set = train_set
        self.alpha = gradient_step_size
        self.epsilon = epsilon
        self.rp = regularization_parameter
        self.steps = steps
        self.n = n_components
        self.m = m
        self.T = T
        self.k = k
    
    @staticmethod
    def _pos_constraint(a):
        indices = np.where(a < 0.0)
        a[indices] = 0.0
        return a

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
            print("DD change is %f and step is %d" %(change,t))

        self.acc_ddsc = acc_ddsc
        self.err_ddsc = err_ddsc
        self.a_ddsc = a_ddsc
        self.b_ddsc = b_ddsc
        return B_cat


    
