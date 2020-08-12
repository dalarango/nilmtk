import numpy as np
import pandas as pd
import time
import librosa
import pickle
from sklearn.decomposition import SparseCoder,DictionaryLearning
from sklearn.metrics import mean_squared_error
from sklearn import cluster
import matplotlib.pyplot as plt


class DSC():

    """
    This is an implementation of the Discriminative Sparse Coding Algorithm proposed by Kolter et al (2010) 
    (http://papers.nips.cc/paper/4054-energy-disaggregation-via-discriminative-sparse-coding.pdf) 

    """

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

    def _initialization(self):
        """
        Initializes the two main arrays needed for decomposition, namely A and B

        Return
        ------
        a: numpy array: initialized signal matrix
        b: numpy array: initialized dictionary

        """
        a = np.random.random((self.n,self.m))
        b = np.random.random((self.T,self.n))
        b /= sum(b)
        return a,b
    
    @staticmethod
    def _pos_constraint(a):
        """
        Ensures non negative values in the signal matrix (A)

        Parameters
        -----------
        a: numpy array

        Return
        ------
        a: numpy array: input array without non-negative values 

        """
        indices = np.where(a < 0.0)
        a[indices] = 0.0
        return a

    def F(self,x,B,A=None):
        '''
        Calculates a gradient-derived matrix (A-prime) based on a known dictionary and a known 
        signal matrix

        Parameters
        -----------
        x : numpy array: base data used for reconstruction
        B : numpy array: Dictionary
        A : numpy array: Singal matrix

        Return
        ------
        acts: numpy array: derived matrix (A-prime)

        '''
        # 4b
        B = np.asarray(B)
        A = np.asarray(A)
        coder = SparseCoder(dictionary=B.T,
                            transform_alpha=self.rp, transform_algorithm='lasso_cd')
        comps, acts = librosa.decompose.decompose(x,transformer=coder)
        acts = self._pos_constraint(acts)

        return acts

    def pre_training(self, no_appliances):
        '''
        This method makes use of Hoyer non-negative sparse coding (alternative algorithm to DSC)
        NOTE: this method is not used as of 08/2020. It is nevetheless included here for 
        future considerations

        Parameters
        -----------
        no_appliances: int: number of appliances to disaggregate with respect to

        return:
        -----------

        A_list: list: base list to construct the signal matrix A
        B_list: list:  base list to construct the dictionary matrix B


        '''

        A_list, B_list = self.nnsc(no_appliances)
        return A_list, B_list

    def nnsc(self, no_appliances):
        '''
        Method as in NNSC from nonnegative sparse coding from P.Hoyer
        NOTE: this method is not used as of 08/2020. It is nevetheless included here for 
        future considerations
        
        Parameters
        -----------
        no_appliances: int: number of appliances to disaggregate with respect to

        return:
        -----------

        A_list: list: base list to construct the signal matrix A
        B_list: list  base list to construct the dictionary matrix B

        '''

        epsilon = 0.01
        A_list = []
        B_list = []
        for x in range(no_appliances):
            A, B = self._initialization()
            Ap = A
            Bp = B
            Ap1 = Ap
            Bp1 = Bp
            t = 0
            change = 1
            while t <= self.steps and self.epsilon <= change:
                Bp = Bp -self.alpha*np.dot((np.dot(Bp,Ap) - x), Ap.T)
                Bp = self._pos_constraint(Bp)
                Bp /= sum(Bp)
                dot2 = np.divide(np.dot(Bp.T, x), (np.dot(np.dot(Bp.T, Bp), Ap) + self. rp))
                Ap = np.multiply(Ap, dot2)
                change = np.linalg.norm(Ap - Ap1)
                change2 = np.linalg.norm(Bp -Bp1)
                Ap1 = Ap
                Bp1 = Bp
                t += 1
                print("NNSC change is %s for iter %s, and B change is %s" %(change,t,change2))
            
            print("Gone through one appliance")
            A_list.append(Ap)
            B_list.append(Bp)
        return A_list, B_list


    def DD(self,x,B,A, real_app_data):
        '''
        Trains a DSC model and computes an optimal dictionary and an optimal singal matrix

        Parameters
        -----------
        x : numpy array: base data used for reconstruction
        B : numpy array: Dictionary
        A : numpy array: Singal matrix'
        real_app_data: numpy array: appliance data to train against

        Return
        ------
        B_cat : numpy array: Optimized dictionary
        theta : float: Scaling parameter

        '''

        #A_star = np.vstack(A)
        #B_cat  = np.hstack(B)
        A_star = A
        B_cat = B
        t = 0
        err = 1
        err_change = 1
        while t <= self.steps and self.epsilon <= err:
            print("Starting iteration .....")
            B_cat_p = B_cat

            acts = self.F(x, B_cat, A=A_star)
            B_cat = (B_cat-self.alpha*((x-B_cat.dot(acts))
                    .dot(acts.T) - (x-B_cat.dot(A_star)).dot(A_star.T)))
            B_cat = self._pos_constraint(B_cat)
            B_cat /= sum(B_cat)
            

            A_prime = self.F(x, B_cat, A=np.vstack(A))

            pred = B_cat.dot(A_prime)
            err_change = err - mean_squared_error(real_app_data, pred)
            err = mean_squared_error(real_app_data, pred)

            change = np.linalg.norm(B_cat - B_cat_p)

            t += 1
            print("DD error change is %f and step is %d" %(err_change, t))
            print("DD current error is %f" %(err))
        t = 0
        err = 1
        theta = 2.0
        gamma = 0.1
        ipsilon = 0.00001
        m = len(real_app_data)
        while t <= self.steps and ipsilon <= change:
            print("Iterating for theta")
            err_p = err 
            A_prime = self.F(x, B_cat, A=np.vstack(A))
            pred = theta * B_cat.dot(A_prime)
            err = mean_squared_error(real_app_data, pred)
            change = abs(err - err_p)
            theta = theta - (1/m) *gamma*(err)
            print("theta is %f" %(theta))
            print("Error change is %f" %(change))
            print(" -------- --------- --------")
        return B_cat, theta




