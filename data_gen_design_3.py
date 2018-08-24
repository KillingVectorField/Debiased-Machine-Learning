import numpy as np
from scipy import stats

def data_design_3(n,p,seed=1):
    '''
    generate data for design 3
    '''
    np.random.seed(seed)
    theta_0 = np.array([1 / j ** 2 for j in range(1,p + 1)])
    gamma_0 = -0.5
    delta = 1
    lambda_00 = 1
    lambda_11 = 2
    alpha = 1
    beta = 0.5

    Sigma = np.zeros((p, p))
    for k in range(p):
        for j in range(p):
            Sigma[k, j] = 0.5 ** (abs(k - j))

    X = np.random.multivariate_normal(np.zeros(p),Sigma,size=(n))

    # Assume Z depends on X_0
    Z = np.random.binomial(1,stats.norm.cdf(X[:,0]))
    
    v = np.random.normal(0,1,size=n)
    D_0 = (gamma_0 + X.dot(theta_0) + v) >= 0
    D_1 = (gamma_0 + X.dot(theta_0) + v + delta) >= 0
    D = Z * D_1 + (1 - Z) * D_0

    xi_3 = np.random.poisson(lambda_11,size=n)
    xi_4 = np.random.poisson(lambda_00,size=n)
    xi_1 = np.random.poisson(np.exp(alpha + X.dot(theta_0)))
    xi_2 = np.random.poisson(np.exp(X.dot(theta_0)))

    tmp = xi_3 * D_1 * D_0 + xi_4 * (D_1 == False) * (D_0 == False)
    Y_1 = xi_1 + tmp
    Y_0 = xi_2 + tmp
    Y = D * Y_1 + (1 - D) * Y_0

    return X,Z,D,Y
