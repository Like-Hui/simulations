import numpy as np
import math
import os
import random
import torch

from stransform import *
from scipy.optimize import fsolve
from scipy.optimize import least_squares

import argparse

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--k', default=60, type=int, help='class number')
parser.add_argument('--n', default=1500, type=int, help='training samples')
parser.add_argument('--beta', default=0.2, type=float, help='n divided by feature dimention p')
parser.add_argument('--seed', default=1111, type=int, help='random seed')
parser.add_argument('--lam', default=1e-5, type=float, help='regularization')
parser.add_argument('--sigma1', default=0, type=float, help='n divided by feature dimention p')
args = parser.parse_args()

k = args.k
n = args.n

beta = args.beta
p = int(n/beta)
var_w = 1
lam = args.lam

print('beta = ', beta)
print('n = ', n)
print('lambda = ', lam)
seeds = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 100]

Rs = 10 * np.arange(1, 31)
Rs = np.insert(Rs, 0, 1)

sigma1 = args.sigma1
acc_final = []
kappa_final = []
tau_final = []

for i in range(len(Rs)):
    accs_seed = []
    kappa_seed = []
    tau_seed = []
    gamma1_seed = []
    gamma2_seed = []
    for seed in seeds:
        os.environ['PYTHONHASHSEED'] = str(seed)
        # Torch RNG
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Python RNG
        np.random.seed(seed)
        random.seed(seed)

        X_train = np.random.randn(n, p) / np.sqrt(p)
        X_train = torch.tensor(X_train)
        U, s, Vh = np.linalg.svd(X_train)
        s = torch.tensor(s)
        U = torch.tensor(U)  # nxn
        Vh = torch.tensor(Vh)  # pxp

        S_tr = s * torch.eye(min(n, p))

        if beta < 1:
            zeros = torch.zeros(min(n, p), max(n, p) - min(n, p))
            S = torch.cat((S_tr, zeros), 1)  # nxp
        else:
            zeros = torch.zeros(max(n, p) - min(n, p), min(n, p))
            S = torch.cat((S_tr, zeros), 0)  # nxp

        W = np.sqrt(var_w) * torch.randn(p, k)
        W_tilta = Vh @ W.double()  # pxk
        sigma_tilta = torch.var(W_tilta)
        Z = X_train @ W.double()

        Z = Z + sigma1 * torch.randn_like(Z) # sigma1 = 0 means no noise
        Y_train = Z.argmax(axis=1)
        var_z = torch.var(Z)

        s_test = np.random.randn(min(n, p))
        s_test = torch.tensor(s_test)
        s1 = s_test * torch.eye(min(n, p))

        if beta < 1:
            S_test = torch.cat((s1, zeros), 1)  # nxp
        else:
            S_test = torch.cat((s1, zeros), 0)  # nxp
            s_test = torch.cat((s_test, torch.zeros(max(n,p)-min(n,p))))

        n_test = n
        X_test = U @ S_test @ Vh # Test samples has different eigenvalues from train samples

        Z_test = X_test @ W.double() # n_test x k
        Y_test = Z_test.argmax(axis=1)

        J = torch.zeros(n_test, k)
        Jhat = torch.zeros(n_test, k)
        J = S_test @ W_tilta  # nxk

        args = [sigma_tilta, var_z, k, beta, Rs[i], lam]
        solution = fsolve(func, [1,1,1,1], args=args) # get kappa, tau, gamma1, gamma2 from fsolve
        print('solution = ', solution)
        if solution[0] < 0 or solution[1] < 0 or solution[2]<0 or solution[3]<0:  # avoid negative solutions
            solution_1 = least_squares(func, [1, 1, 1, 1], args=(args,), jac='3-point', bounds=([0, 0, 0, 0], np.inf),
                                     max_nfev=8000)
            solution = solution_1.x
            print(solution_1)

        kappa = solution[0]**2
        tau = solution[1]**2
        gamma1 = solution[2]
        gamma2 = lam/solution[-1]

        Q = J + np.sqrt(tau) * torch.randn(n_test,k)
        Jhat = (s_test**2)[:, None]*Q/((s_test**2)[:, None]+lam/gamma2)

        z_test_ =  np.maximum(0, U @ Jhat)
        y_pred = z_test_.argmax(axis=1)

        acc_test = torch.sum(y_pred == Y_test)/n_test * 100

        accs_seed.append(acc_test)
        kappa_seed.append(kappa)
        tau_seed.append(tau)
        gamma1_seed.append(gamma1)
        gamma2_seed.append(gamma2)

    print('kappa = %.4f' % (sum(kappa_seed)/len(kappa_seed)))
    print('tau = %.4f' % (sum(tau_seed)/len(tau_seed)))
    print('gamma+ = %.4f' % (sum(gamma1_seed)/len(gamma1_seed)))
    print('gamma- = %.6f' % (sum(gamma2_seed)/len(gamma2_seed)))
    print('accuracy of 10 seeds = ', accs_seed)
    print('k = %d, R = %d, 10 seeds average accuracy: %.2f%%' % (k, Rs[i], (sum(accs_seed)/len(accs_seed))))

    kappa_ave = sum(kappa_seed) / len(kappa_seed)
    tau_ave = sum(tau_seed) / len(tau_seed)
    acc_ave = sum(accs_seed) / len(accs_seed)

    acc_final.append(acc_ave)
    kappa_final.append(kappa_ave)
    tau_final.append(tau_ave)
np.save('results/accs_beta' + str(beta) + '_k' + str(k) + '_n' + str(n) + '.npy', acc_final)
np.save('results/kappa_beta' + str(beta) + '_k' + str(k) + '_n' + str(n) + '.npy', kappa_final)
np.save('results/tau_beta' + str(beta) + '_k' + str(k) + '_n' + str(n) + '.npy', tau_final)



