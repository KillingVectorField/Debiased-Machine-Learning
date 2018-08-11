## A simple simulation for DML estimators that use 5 machine learning methods(default settings) to train the nuisance parameters
# %%import
from __future__ import print_function
import numpy as np
import math
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
# %% DML estimator
def DML(y,d,x,K=2,DML=2,method='LinearRegression',alpha=0.05):
    n = x.shape[0]
    p = x.shape[1]
    lst = range(n)
    random.shuffle(lst)
    index = np.array_split(lst,K)
    theta_check = np.ones(K)
    theta_hat = np.ones(K)
    slope = np.ones(K)
    intercept = np.ones(K)
    slope0 = np.ones(K)
    intercept0 = np.ones(K)
    for k in range(K):
        auxiliary_part = index[k]
        main_part = np.delete(range(n),index[k])
        if method=='LinearRegression':
            modell = LinearRegression()
        if method=='GradientBoostingRegressor':
            modell = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=100,subsample=0.7,min_samples_split=2,min_samples_leaf=1,max_depth=10,init=None,random_state=None,max_features=None,alpha=0.9,verbose=0,max_leaf_nodes=None,warm_start=False)
        if method=='RandomForestRegressor':
            modell = RandomForestRegressor(oob_score=True, n_jobs=-1, random_state=50, max_features="auto")
        if method=='Lasso':
            modell = Lasso(alpha=0.5)
        if method=='MLPRegressor':
            modell = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=None)
        modell.fit(x[auxiliary_part], d[auxiliary_part])
        V_hat = d[main_part]-modell.predict(np.array(x[main_part]))
        modell.fit(x[auxiliary_part], y[auxiliary_part])
        g0_hat = modell.predict(np.array(x[main_part]))

        # original estimator using the original score function
        slope0[k] = np.mean(np.power(d[main_part], 2))
        intercept0[k] = np.mean(d[main_part].dot((y[main_part]-g0_hat)))
        theta_hat[k] = np.power(slope[k],-1)*intercept[k]

        # DML estimator using the Neyman orthogonal score function
        slope[k] = np.mean(V_hat.dot(d[main_part]))
        intercept[k] = np.mean(V_hat.dot(y[main_part]-g0_hat))
        theta_check[k] = np.power(slope[k],-1)*intercept[k]
    if DML==1:
        # DML1 estimator
        theta_tilde = np.mean(theta_check)
    if DML==2:
        # DML2 estimator
        theta_tilde = np.mean(intercept)/np.mean(slope)

    # original estimator using the original score function for comparison
    theta_original = np.mean(intercept0)/np.mean(slope0)
    # estimating variance
    J0_hat = np.mean(slope)
    sigma2_hat = (1/J0_hat)*np.mean(np.power((intercept-(theta_tilde*slope)),2))*(1/J0_hat)
    CI = [theta_tilde-norm.ppf(1-alpha/2)*np.power(sigma2_hat/n,0.5),theta_tilde+norm.ppf(1-alpha/2)*np.power(sigma2_hat/n,0.5)]
    # print("original estimate:",end = "")
    # print(theta_original)
    # print("theta:",end="")
    # print(theta_tilde,end="  ")
    # print("CI:",end="")
    # print(CI)
    return (theta_tilde,CI,sigma2_hat,theta_original)
# %%Data Generation
np.random.seed(0)
n = 200;
p = 250
v = np.random.normal(0, 1, n)
u = np.random.normal(0, 1, n)
Sigma = np.ones((p, p))
for j in range(p):
    for k in range(p):
        Sigma[k,j] = np.power(0.5,abs(j-k))
x = np.random.randn(n, p).dot(Sigma)

theta0 = np.ones(p)
for j in range(p):
    theta0[j] = np.power(1.0/(j+1),2)
Rset = [0,0.1,0.5,0.9]
Rd2 = Rset[1]
Ry2 = Rset[1]
cd = np.power((np.power(math.pi,2)/3)*Rd2/((1-Rd2)*np.transpose(theta0).dot(Sigma).dot(theta0)),0.5)
cy = np.power(Ry2/((1-Ry2)*np.transpose(theta0).dot(Sigma).dot(theta0)),0.5)
d = np.ones(n)
y = np.ones(n)
for i in range(n):
    d[i] = np.exp(np.transpose(x[i]).dot(cd*theta0))/(1+np.exp(np.transpose(x[i]).dot(cd*theta0)))+v[i]
    y[i] = d[i]+x[i].T.dot(cd*theta0)+u[i]
#%% Simulation
# K=5 DML1
K=5
methodname = ['LinearRegression','GradientBoostingRegressor','RandomForestRegressor','Lasso','MLPRegressor']
result = np.ones((5, 100))
var = np.ones((5, 100))
median_results = np.ones(5)
median_CI = np.ones((5,2))
for j in range(5):
    for i in range(100):
        out = DML(y,d,x,K,DML=1,method=methodname[j],alpha=0.05)
        result[j,i] = out[0]
        var[j, i] = out[2]
    median_results[j] = np.median(result[j])
    median_variance = np.median(var[j])
    median_CI[j] = [median_results[j]-norm.ppf(1-0.05/2)*np.power(median_variance,0.5),median_results[j]+norm.ppf(1-0.05/2)*np.power(median_variance,0.5)]
print(median_results)
print(median_CI)
# boxplot
plt.boxplot(x = [result[0],result[1],result[2],result[3],result[4]],
            patch_artist=True,
            labels = ['Linear','GradientBoosting','RandomForest','Lasso','MLP'],
            showmeans=True,
            boxprops = {'color':'black','facecolor':'#9999ff'},
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
            meanprops = {'marker':'D','markerfacecolor':'indianred'},
            medianprops = {'linestyle':'--','color':'orange'})
plt.savefig('DML1 K=5.png',dpi=600)
plt.show()