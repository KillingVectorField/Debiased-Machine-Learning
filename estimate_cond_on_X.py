import numpy as np
from data_gen_design_3 import *

data = data_design_3(1000000,1,seed=1,cond=True)
X,Z,D,Y = data

mu_00 = np.mean(np.extract((D == 0) * (Z == 0),Y))
mu_01 = np.mean(np.extract((D == 0) * (Z == 1),Y))
mu_10 = np.mean(np.extract((D == 1) * (Z == 0),Y))
mu_11 = np.mean(np.extract((D == 1) * (Z == 1),Y))

pr_at = np.sum((Z == 0) * (D == 1)) / np.sum(Z == 0)#always taker
pr_nt = np.sum((Z == 1) * (D == 0)) / np.sum(Z == 1)#never taker
pr_co = 1 - pr_at - pr_nt

print('pr_at =',pr_at)
print('pr_co =',pr_co)
print('pr_nt =',pr_nt)
#subgroup proportion is unbiasedly estimated


hat_lambda_00 = (mu_01 - mu_00) / (pr_co / (pr_nt + pr_co))
hat_lambda_11 = (mu_10 - mu_11) / (pr_co / (pr_at + pr_co))

print('hat_lambda_11:', hat_lambda_11)
print('hat_lambda_00:', hat_lambda_00)

hat_alpha = np.log((mu_10 - hat_lambda_11) / (mu_01 - hat_lambda_00))
print('hat_alpha:',hat_alpha)
#alpha is consistenly estimated when n is large enough