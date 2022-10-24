import numpy as np
from binorm_cdf import bivariate_normal_cdf
relu = lambda x: np.maximum(0, x)

K = 3
n_samples = 10000
rho = 0.9

X1, X2 = np.split(np.random.multivariate_normal(
    np.zeros(2*K), 
    np.vstack([
        np.hstack([np.eye(K),np.eye(K)*rho]),
        np.hstack([np.eye(K)*rho,np.eye(K)])
        ]),
    n_samples
), 2, axis=1)
y1 = np.argmax(relu(X1), axis=1)
y2 = np.argmax(relu(X2), axis=1)
accuracy = np.sum(y1==y2)/len(y1)

print(accuracy)
x1,x2 = np.split(np.random.multivariate_normal(
    np.zeros(2), np.array([[1, rho],[rho, 1]]), n_samples
),2,axis=1)
print((bivariate_normal_cdf(x1,x2,rho)**(K-1)).mean()*K)