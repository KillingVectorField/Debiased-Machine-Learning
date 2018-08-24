import numpy as np
from scipy import stats


gamma_0 = -0.5
delta = 1
lambda_00 = 1
lambda_11 = 2
alpha = 1
beta = 0.5

def data_design_3(n,p,seed=1,cond=True,linear=False):
    '''
    generate data for design 3
    '''
    theta_0 = np.array([1 / j ** 2 for j in range(1,p + 1)])

    np.random.seed(seed)

    Sigma = np.zeros((p, p))
    for k in range(p):
        for j in range(p):
            Sigma[k, j] = 0.5 ** (abs(k - j))

    # Simulate Conditional on X
    if(cond):
        X = np.array([np.random.multivariate_normal(np.zeros(p),Sigma)])
        X = np.repeat(X,repeats=n,axis=0)
    else:
        X = np.random.multivariate_normal(np.zeros(p),Sigma,size=(n))

    # Assume Z depends on X_0
    Z = np.random.binomial(1,stats.norm.cdf(X[:,0]))
    
    v = np.random.normal(0,1,size=n)
    D_0 = (gamma_0 + X.dot(theta_0) + v) >= 0
    D_1 = (gamma_0 + X.dot(theta_0) + v + delta) >= 0
    D = Z * D_1 + (1 - Z) * D_0
    pr_at=sum((D_0 == 1) * (D_1 == 1)) / n
    pr_co=sum((D_0 == 0) * (D_1 == 1)) / n
    pr_df=sum((D_0 == 1) * (D_1 == 0)) / n
    pr_nt=sum((D_0 == 0) * (D_1 == 0)) / n
    print('pr_at =',pr_at)
    print('pr_co =',pr_co)
    print('pr_df =',pr_df) # no defier
    print('pr_nt =',pr_nt)
    print('cor(Z,D_1):', np.corrcoef(Z,D_1))#seems independent
    print('cor(Z,D_0):', np.corrcoef(Z,D_0))#seems independent

    print(sum((D_1 == 1) * (D_0 == 1) * (D == 1) * (Z == 1)) / sum((D == 1) * (Z == 1)), "should equal pr_at/(pr_at+pr_co) =", pr_at/(pr_at+pr_co))
    print(sum((D_1 == 1) * (D_0 == 1) * (D == 1) * (Z == 0)) / sum((D == 1) * (Z == 0)),"should equal 1.0")
    print(sum((D_1 == 0) * (D_0 == 0) * (D == 0) * (Z == 0)) / sum((D == 0) * (Z == 0)), "should equal pr_nt/(pr_nt+pr_co) =", pr_nt/(pr_nt+pr_co))
    print(sum((D_1 == 0) * (D_0 == 0) * (D == 0) * (Z == 1)) / sum((D == 0) * (Z == 1)), "should equal 1.0")

    if(linear):
        pass
    else:
        xi_3 = np.random.poisson(lambda_11,size=n)
        xi_4 = np.random.poisson(lambda_00,size=n)
        print(np.mean(xi_3),np.mean(xi_4))
        xi_1 = np.random.poisson(np.exp(alpha + X.dot(theta_0)))
        xi_2 = np.random.poisson(np.exp(X.dot(theta_0)))

        tmp = xi_3 * D_1 * D_0 + xi_4 * (D_1 == False) * (D_0 == False)
        Y_1 = xi_1 + tmp
        Y_0 = xi_2 + tmp
        Y = D * Y_1 + (1 - D) * Y_0

    return X,Z,D,Y