#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:38:15 2018

@author: WangJianqiao
"""

import numpy as np
from math import sqrt, pi
def dataset(n, p, interaction=None):
    '''
    generate data for simulation, returns X, D, Y
    '''
    # covariance for X
    cov = np.zeros((p, p))
    for k in range(p):
        for j in range(p):
            cov[k, j] = 0.5 ** (abs(k - j))
            
    # generate noise U, V
    V = np.random.normal(loc=0, scale=1, size=(n, 1))
    U = np.random.normal(loc=0, scale=1, size=(n, 1))

    # theta
    theta_0 = np.mat([(1 / j) ** 2 for j in range(1, p + 1)]).T
    
    # c_d and c_y
    # Rd and Ry: {0, 0.1, 0.5, 0.9}
    Rd = 0.5
    Ry = 0.5
    c_d = sqrt(((pi ** 2 / 3) * (Rd ** 2)) / ((1 - Rd ** 2) \
        * np.matmul(np.matmul(theta_0.T, cov), theta_0)[0, 0]))
    c_y = sqrt((Ry ** 2) / ((1 - Ry ** 2) \
        * np.matmul(np.matmul(theta_0.T, cov), theta_0)[0, 0]))
    
    # generate X
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=(n))

    # generate D
    D = np.exp(np.matmul(X, c_d * theta_0)) \
    / (1 + np.exp(np.matmul(X, c_d * theta_0))) + V
    
    # generate Y, depends on interaction
    if interaction is None:
        Y = D + np.matmul(X, c_y * theta_0) + U
    else:
        Y = D + np.matmul(X, c_y * theta_0) \
        + np.multiply(D, X[:, interaction].sum(axis=1).reshape((n, 1))) \
        + U
    return X, D, Y
