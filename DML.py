#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:41:53 2018

@author: WangJianqiao
"""
import numpy as np
from math import sqrt
from sklearn.model_selection import KFold
from scipy.stats import norm
from math import pi

class Partial_Linear():
    '''
    DML estimator for partial linear model
    '''
    def __init__(self, K, S, estimator_Y, estimator_D, mode='median'):
        self.K = K # integer, for K-fold random partition of observations
        self.S = S # integer, repeat DML estimator S times
        self.estimator_Y = estimator_Y # machine learning estimator for Y
        self.estimator_D = estimator_D # machine learning estimator for D
        self.DML1 = [] # list for S DML1 estimors
        self.DML2 = [] # list for S DML2 estimors
        self.mode = mode # string, two modes: mean or median
    
    def fit(self, X, D, Y, interaction=None):
        '''
        X: n x p matrix
        D: n x 1 vector
        Y: n x 1 vector
        interaction: list, a list of interaction terms' index; 
                     if interaction is None, we do not consider interaction terms 
                     in partial linear model
        if interactions are considered:
            parameters of interest: theta
        if interaction are not considered:
            parameters of interest: (theta, eta1, eta2, eta3, eta4,...).T
        '''
        self.interaction = interaction
        self.X, self.D, self.Y = X, D, Y
        self.n, self.p = X.shape # n: number of samples; p: number of features
        self.sigma_square_DML1 = [] # list for S DML1 estimation of variance
        self.sigma_square_DML2 = [] # list for S DML2 estimation of variance
        self.kf = KFold(n_splits=self.K) # for sample splitting

        if self.interaction is None:
            '''
            partial linear model without interaction terms
            '''

            # S times DML
            for s in range(self.S):
                self._train()
                
            self.DML1 = np.array(self.DML1)
            self.DML2 = np.array(self.DML2)
            self.sigma_square_DML1 = np.array(self.sigma_square_DML1)
            self.sigma_square_DML2 = np.array(self.sigma_square_DML2)
            
            # adjusted estimators and confidence interval by mean or median
            if self.mode == 'mean':
                # DML1 point estimate of parameters of interest
                self.theta_DML1 = np.mean(self.DML1) 

                # DML2 point estimate of parameters of interest
                self.theta_DML2 = np.mean(self.DML2) 
                
                # DML1 estimate of variance of parameters of interest
                self.sigma_square1 = np.mean(self.sigma_square_DML1 + (self.DML1 - self.theta_DML1) ** 2)
                
                # DML2 estimate of variance of parameters of interest
                self.sigma_square2 = np.mean(self.sigma_square_DML2 + (self.DML2 - self.theta_DML2) ** 2)
            
            elif self.mode == 'median':
                # DML1 point estimate of parameters of interest
                self.theta_DML1 = np.median(self.DML1)
                
                # DML2 point estimate of parameters of interest
                self.theta_DML2 = np.median(self.DML2)
                
                # DML1 estimate of variance of parameters of interest
                self.sigma_square1 = np.median(self.sigma_square_DML1 + (self.DML1 - self.theta_DML1) ** 2)
                
                # DML2 estimate of variance of parameters of interest
                self.sigma_square2 = np.median(self.sigma_square_DML2 + (self.DML2 - self.theta_DML2) ** 2)
        else:
            '''
            partial linear model considering interactions
            '''
            # self.A concatenate D and D(X*)
            self.A = np.concatenate([self.D, np.multiply(self.D, self.X[:, self.interaction])], axis=1)
            
            # S times DML
            for s in range(self.S):
                self._train()

            # adjusted estimators and confidence interval by mean or median
            if self.mode == 'mean':
                # DML1 point estimate of parameters of interest
                self.theta_DML1 = np.mean(self.DML1, axis=1).reshape((self.DML1.shape[0], 1))
                
                # DML2 point estimate of parameters of interest
                self.theta_DML2 = np.mean(self.DML2, axis=1).reshape((self.DML2.shape[0], 1))
                
                self.sigma_square1 = 0
                self.sigma_square2 = 0
                for i in range(self.S):
                    self.sigma_square1 += self.sigma_square_DML1[i] + np.matmul(self.DML1[:, i].reshape(self.theta_DML1.shape) - self.theta_DML1, (self.DML1[:, i].reshape(self.theta_DML1.shape) - self.theta_DML1).T)
                    self.sigma_square2 += self.sigma_square_DML2[i] + np.matmul(self.DML2[:, i].reshape(self.theta_DML2.shape) - self.theta_DML2, (self.DML2[:, i].reshape(self.theta_DML2.shape) - self.theta_DML2).T)
                
                # DML1 estimate of covariance of parameters of interest
                self.sigma_square1 = self.sigma_square1 / self.S
                
                # DML2 estimate of covariance of parameters of interest
                self.sigma_square2 = self.sigma_square2 / self.S
            
            elif self.mode == 'median':
                # DML1 point estimate of parameters of interest
                self.theta_DML1 = np.median(self.DML1, axis=1).reshape((self.DML1.shape[0], 1))
                
                # DML2 point estimate of parameters of interest
                self.theta_DML2 = np.median(self.DML2, axis=1).reshape((self.DML2.shape[0], 1))
                
                sigma1_list = []
                sigma2_list = []
                for i in range(self.S):
                    sigma1_list.append(self.sigma_square_DML1[i] + np.matmul(self.DML1[:, i].reshape(self.theta_DML1.shape) - self.theta_DML1, (self.DML1[:, i].reshape(self.theta_DML1.shape) - self.theta_DML1).T))
                    sigma2_list.append(self.sigma_square_DML2[i] + np.matmul(self.DML2[:, i].reshape(self.theta_DML2.shape) - self.theta_DML2, (self.DML2[:, i].reshape(self.theta_DML2.shape) - self.theta_DML2).T))
                sigma1_list = np.array(sigma1_list)
                sigma2_list = np.array(sigma2_list)
                
                # DML1 estimate of covariance of parameters of interest
                self.sigma_square1 = np.median(sigma1_list, axis=0)
                
                # DML2 estimate of covariance of parameters of interest
                self.sigma_square2 = np.median(sigma2_list, axis=0)

    def _train(self):
        '''
        one-time DML estimation, without mean or median adjustments
        '''
        DML2_1 = 0 # the first term of the DML2 estimator
        DML2_2 = 0 # the second term of the DML2 estimator
        theta_DML1_sum = 0 # sum of DML1 estimator for theta and finally average it
        
        # used for computing DML1 or DML2 confidence interval
        if self.interaction is None:
            sigma_matrix1 = np.matlib.zeros((self.n // self.K, self.K)) 
            sigma_matrix2 = np.matlib.zeros((self.n // self.K, self.K)) 
        else:
            # interactions are considered
            residual_A_matrix = np.mat(np.zeros(self.A.shape))
            residual_Y_matrix = np.mat(np.zeros(self.Y.shape))
        
        k = 0
        np.random.shuffle(self.X)

        for nuisance, interest in self.kf.split(self.X):
            # dataset for estimating nuisance parameters
            X_nuisance, Y_nuisance, D_nuisance = self.X[nuisance], self.Y[nuisance], self.D[nuisance]
            # dataset for estimating parameters of interest
            X_interest, Y_interest, D_interest = self.X[interest], self.Y[interest], self.D[interest]
            if self.interaction is not None:
                A_nuisance, A_interest = self.A[nuisance], self.A[interest]
            
            # estimate nuisance parameters
            self.estimator_Y.fit(X_nuisance, Y_nuisance)
            self.estimator_D.fit(X_nuisance, D_nuisance)
            
            # residual for Y_interest and D_interest
            predict_Y = self.estimator_Y.predict(X_interest).reshape(Y_interest.shape)
            predict_D = self.estimator_D.predict(X_interest).reshape(D_interest.shape)
            residual_Y = Y_interest - predict_Y
            residual_D = D_interest - predict_D
            if self.interaction is not None:
                # residual for A_interest when interactions are considered
                predict_A = np.concatenate([predict_D, np.multiply(predict_D, X_interest[:, self.interaction])], axis=1)
                residual_A = A_interest - predict_A
            
            if self.interaction is None:
                # DML1 estimator of parameter of interest
                theta_DML1_sum += np.sum(np.multiply(residual_D, residual_Y)) / np.sum(np.multiply(residual_D, residual_D))
                # DML2 estimator of parameter of interest
                DML2_1 += np.sum(np.multiply(residual_D, residual_Y))
                DML2_2 += np.sum(np.multiply(residual_D, residual_D))
                
                # terms for computing confidence interval
                sigma_matrix1[:, k] = np.multiply(residual_D, residual_Y)
                sigma_matrix2[:, k] = np.multiply(residual_D, residual_D)
            
            else: 
                # interactions are considered
                P = 0 # sum of (residual_Yi * residual_Ai), for later estimation
                Q = 0 # sum of (residual_Ai * residual_Ai.T), for later estimation
                for i in range(A_interest.shape[0]):
                    residual_Ai = residual_A[i, :].reshape((A_interest.shape[1], 1))
                    residual_Yi = residual_Y[i]
                    Q += np.multiply(residual_Yi, residual_Ai)
                    P += np.matmul(residual_Ai, residual_Ai.T)  
                
                
                theta_DML1_sum += np.matmul(P.I, Q)
                
                # DML2 estimator for parameters of interest
                DML2_1 += Q
                DML2_2 += P
                residual_A_matrix[k * residual_A.shape[0]: (k + 1) * residual_A.shape[0], ] = residual_A
                
                residual_Y_matrix[k * residual_Y.shape[0]: (k + 1) * residual_Y.shape[0], ] = residual_Y

            k += 1

        if self.interaction is None:
            # DML1 and DML2 estimations of parameter of interest
            theta_DML1 = theta_DML1_sum / self.K
            theta_DML2 = DML2_1 / DML2_2
            
            # save the estimations and adjust them by mean or median finally
            self.DML1.append(theta_DML1)
            self.DML2.append(theta_DML2)
            
            # J_0, mat1, mat2: some variables for computing variances
            J_0 = DML2_2 / self.n
            mat1 = sigma_matrix1 - theta_DML1 * sigma_matrix2
            mat2 = sigma_matrix1 - theta_DML2 * sigma_matrix2

            # variances for DML1 and DML2 estimations
            sigma1_square = (J_0 ** -1) * (np.sum(np.multiply(mat1, mat1)) / self.n) * (J_0 ** -1)
            sigma2_square = (J_0 ** -1) * (np.sum(np.multiply(mat2, mat2)) / self.n) * (J_0 ** -1)
            
            # save estimations of variance and adjust them by mean or median finally
            self.sigma_square_DML1.append(sigma1_square)
            self.sigma_square_DML2.append(sigma2_square)
        
        else:
            # DML1 and DML2 estimations of parameter of interest
            theta_DML1 = theta_DML1_sum / self.K
            theta_DML2 = np.matmul(DML2_2.I, DML2_1)
        
            # save the estimations and adjust them by mean or median finally
            if len(self.DML1) == 0:
                self.DML1 = theta_DML1
            else:
                self.DML1 = np.concatenate([self.DML1, theta_DML1], axis=1)
            if len(self.DML2) == 0:
                self.DML2 = theta_DML2
            else:
                self.DML2 = np.concatenate([self.DML2, theta_DML2], axis=1)
            
            # J_0, mat1, mat2: some variables for computing variances
            J_0 = DML2_2 / self.n
            mat1 = 0
            mat2 = 0
            for i in range(residual_A_matrix.shape[0]):
                rAi = residual_A_matrix[i, :].reshape((residual_A_matrix.shape[1], 1))
                rYi = residual_Y_matrix[i]
                ayi = np.multiply(rYi, rAi)
                aai = np.matmul(rAi, rAi.T)
                mat1 += np.matmul(ayi, ayi.T) - np.matmul(ayi, np.matmul(theta_DML1.T, aai)) - np.matmul(ayi, np.matmul(theta_DML1.T, aai)).T + np.matmul(aai, np.matmul(np.matmul(theta_DML1, theta_DML1.T), aai))
                mat2 += np.matmul(ayi, ayi.T) - np.matmul(ayi, np.matmul(theta_DML2.T, aai)) - np.matmul(ayi, np.matmul(theta_DML2.T, aai)).T + np.matmul(aai, np.matmul(np.matmul(theta_DML2, theta_DML2.T), aai))
            
            # variances for DML1 and DML2 estimations
            sigma1_square = np.matmul(J_0.I, np.matmul(mat1 / self.n, (J_0.I).T))
            sigma2_square = np.matmul(J_0.I, np.matmul(mat2 / self.n, (J_0.I).T))
            
            # save estimations of variance and adjust them by mean or median finally
            self.sigma_square_DML1.append(sigma1_square)
            self.sigma_square_DML2.append(sigma2_square)

        

    def coef(self):
        '''
        return DML estimate of parameter of interest
        '''
        return self.theta_DML1, self.theta_DML2
    
    def confidence_interval(self, alpha=0.95, i=None):
        '''
        Confidence interval estimator
        i is used when interactions are considered, and i denotes the ith parameter of interest
        if i is None, return confidence interval estimation of the first parameter of interest
        '''
        self.alpha = alpha
        if self.interaction is None:
            # No interactions, only return confidence interval for theta
            
            # DML1 confidence interval of parameters of interest
            self.CI1 = tuple(self.theta_DML1 \
                             + sqrt(self.sigma_square1 / self.n) \
                             * np.array(norm.interval(self.alpha)))
            
            # DML2 confidence interval of parameters of interest
            self.CI2 = tuple(self.theta_DML2 \
                             + sqrt(self.sigma_square2 / self.n) \
                             * np.array(norm.interval(self.alpha)))
        else:
            # Consider interactions
            if i is None:
                return self.confidence_interval(alpha=self.alpha, i=0)
            
            # DML1 confidence interval of the ith parameters of interest
            ci1 = self.theta_DML1[i] \
                  + sqrt(float(self.sigma_square1[i, i]) \
                  / float(self.n)) * np.array(norm.interval(self.alpha))
            
            # DML2 confidence interval of the ith parameters of interest
            ci2 = self.theta_DML2[i] \
                  + sqrt(float(self.sigma_square2[i, i]) \
                  / float(self.n)) * np.array(norm.interval(self.alpha))
            
            self.CI1 = (ci1[0, 0], ci1[0, 1])
            self.CI2 = (ci2[0, 0], ci2[0, 1])
        
        return self.CI1, self.CI2
        
        