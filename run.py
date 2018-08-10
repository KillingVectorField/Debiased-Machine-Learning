#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 19:13:45 2018

@author: WangJianqiao
"""

from Dataset import dataset
from DML import Partial_Linear
from sklearn.linear_model import Lasso


n = 200
p = 250
# K = 2
K = 2
S = 10
estimator_Y = Lasso(alpha=0.01)
estimator_D = Lasso(alpha=0.01)
# no interaction
X, D, Y = dataset(n, p)
partial_linear = Partial_Linear(K, S, estimator_Y, estimator_D, mode='mean')
partial_linear.fit(X, D, Y)
theta_DML1, theta_DML2 = partial_linear.coef()
CI_DML1, CI_DML2 = partial_linear.confidence_interval(alpha=0.95)

# interaction
X, D, Y = dataset(n, p, interaction=[0, 1])
partial_linear = Partial_Linear(K, S, estimator_Y, estimator_D, mode='mean')
partial_linear.fit(X, D, Y, interaction=[0, 1])
theta_DML1, theta_DML2 = partial_linear.coef()
theta_CI_DML1, theta_CI_DML2 = partial_linear.confidence_interval(alpha=0.95, i=[0])
