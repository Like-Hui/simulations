import numpy as np 
import matplotlib.pyplot as plt
from binorm_cdf import bivariate_normal_cdf

n_samples = 10000
K_array=[2, 5, 10, 15, 20]
rho_array = np.linspace(0.01, 0.95, 10)
accuracy = np.zeros((len(K_array), len(rho_array)))
for j, rho in enumerate(rho_array):
    x1, x2 = np.split(np.random.multivariate_normal(
            np.zeros(2), np.array([[1, rho],[rho, 1]]), n_samples
        ), 2, axis=1)
    phi = bivariate_normal_cdf(x1, x2, rho)
    accuracy[:, j] = [(phi**(K-1)).mean()*K for K in K_array]
    print(rho)
for i, K in enumerate(K_array): plt.plot(rho_array, accuracy[i], label=K)
plt.legend()
plt.savefig('accuracy.png')

