import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc
import numpy as np
import torch.nn.functional as F
import os
import random

from torch.nn.functional import one_hot

# (1-beta)^2 tau^2 = E|zhat-beta*p-(1-beta)z*|^2 
# set z*, get Y
# set beta, p, together with Y get zhat, then can get tau

import scipy.optimize

# # def fun(x):
# #         gamma*(2*lim+1-lim+lim*lamda/gamma_1-np.sqrt((1-lim+lim*lamda/gamma_1)**2+4*lim*lamda/gamma_1))-np.sqrt((1-lim+lim*lamda/gamma_1)**2+4*lim*lamda/gamma_1)*gamma_1 + (1-lim+lim*lamda/gamma_1)*gamma_1=0
# #         return gamma_1

# def solve_function(unsolved_value):
#     gamma 1=unsolved_value[0]
#     return [
#         7.1162*gamma_1**2-9.44*gamma_1 +0.9    ]

seed = 3333
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

n = 5000
k = 100
sigma=1
sigma1 = 0.3
kappa = sigma1**2

gamma = 5
R = 20

Z = sigma*torch.randn(n, k)
y = Z.argmax(axis=1)
Y = one_hot(Z.argmax(axis=1), k).float()
P = Z + sigma1 * torch.randn_like(Z)

p_1 = R*(1-np.sqrt(1+1.0/gamma))

erf = math.erf(-1.0 * p_1/np.sqrt(2*sigma**2+2*kappa))

beta = 0.01
# beta = -0.5*erf/k/(1+gamma) + (2*gamma+1)/2/(1+gamma)
print('beta = ', beta)

zhat = torch.ones(n, k)

index_1=Z.argmax(axis=1)

for i in np.arange(n):
  for j in np.arange(k):
    if P[i,j] > 0:
      zhat[i,j] = gamma*P[i,j]/(1+gamma)
    else:
      zhat[i,j] = P[i,j]

t=0

for i in np.arange(n):
  if P[i, index_1[i]]>p_1:
    zhat[i, index_1[i]] = (R+gamma*P[i, index_1[i]])/(1+gamma)
  else:
    zhat[i, index_1[i]] = P[i, Z.argmax(axis=1)[i]]

  t += (zhat[i, index_1[i]]-beta*P[i, index_1[i]]-(1-beta)*Z[i, index_1[i]])**2
  
t_1 = t/n  # t_1 corresponds to the estimation of y=1

t2 = torch.mean((zhat-beta*P-(1-beta)*Z)**2)
t_0 = (torch.sum((zhat-beta*P-(1-beta)*Z)**2)/n - t_1)/(k-1) # t_0 corresponds to the estimation of y=0


t_0 = t_0 / (1-beta) / (1-beta)
t_1 = t_1 / (1-beta) / (1-beta)
t2 = t2 / (1-beta) / (1-beta)

a = 1/(1+gamma)/(1+gamma) * (R**2+sigma**2+(gamma-beta-beta*gamma)**2*kappa+2*R*(gamma-beta-beta*gamma)*np.sqrt(kappa)-2*R*np.sqrt(2*np.log(k))*sigma)
b = (1-beta)**2*kappa

t_1_est = (0.5+0.5*erf)*a + (0.5-0.5*erf)*b

t_1_est = t_1_est/(1-beta) /(1-beta)

e = 1/(1+gamma)/(1+gamma) * (sigma**2+(gamma-beta-beta*gamma)**2*kappa)


print('erf = ', erf)

print('t_1, t_1_est = ', t_1, t_1_est)

t_0_est = 0.5*e + 0.5*b
t_0_est = t_0_est / (1-beta) / (1-beta)

print('t_0, t_0_est = ', t_0, t_0_est)

t_est = (t_1_est + (k-1)*t_0_est)/k

print('tau, tau_est = ', t2, t_est)
