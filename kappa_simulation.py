from numpy.lib.index_tricks import diag_indices
import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gc
import numpy as np
import torch.nn.functional as F
import os
import random

from torch.nn.functional import one_hot

from torchvision import transforms

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
gamma = 50
lamda = 0.0001
delta = 100
sigma = 80
tau = 5

A = np.random.randn(n, k)
U, s, Vh = np.linalg.svd(A)

s=torch.tensor(s)
ss = s**2
ss_mean = torch.mean(ss)
ss = ss * delta / ss_mean

s = torch.sqrt(ss)
print('ss mean = ', torch.mean(s**2))

w = np.sqrt(sigma)*torch.randn(n,k)
J = s*w

Q = J + np.sqrt(tau) * torch.randn_like(J)

jhat = gamma*s*s*Q/(gamma*s*s+lamda)

grad = 1-lamda/gamma/(delta-1)

print('grad = ', grad)
tmp=(jhat-grad*Q)/(1-grad)-J
print('tmp shape = ', tmp.shape)
print('shape of diag = ', torch.diag(tmp.T @ tmp).shape)

# kappa = torch.diag(tmp.T @ tmp).median()/n
# kappa=torch.median(tmp**2, axis=0)

kappa = torch.mean(tmp**2)
print('kappa expectation = ', kappa)

kappa_1 = 1.0/(delta-1)*tau+(delta-1)*sigma
print('kappa estimated = ', kappa_1)
