import numpy as np
import math

def T(u, beta):
    return (-(1-beta+u)+np.sqrt((1-beta+u)**2+4*beta*u))/2/beta

def T_prime(u, beta):
    return (-1+(1+beta+u)/np.sqrt((1-beta+u)**2+4*beta*u))/2/beta

def T3(u, beta):
    return T(u, beta) - u*T_prime(u, beta)

def T4(u, beta):
    return 1 - T(u, beta) - u*T_prime(u, beta)


def func(variables, args):
    (kappa, tau, gamma, u) = variables
    sigma_tilta = args[0]
    var_z = args[1]
    k = args[2]
    beta = args[3]
    R = args[4]
    lam = args[5]
    return [lam / u * T(u, beta) / (1 - T(u, beta)) - gamma,
            (-math.erf(R * (1 - np.sqrt(1 + 1.0 / gamma))) + k * (2 * gamma + 1)) /2/k/(1 + gamma) - T(u, beta),
            (tau * tau * (T4(u, beta) - (1 - T(u, beta)) ** 2) + sigma_tilta ** 2 * u * u * T_prime(u, beta)) / T(u,
                                                                                                                  beta) / T(
                u, beta) - kappa ** 2,
            0.5 * kappa * kappa * (k ** 2 + k * math.erf(R * (1 - np.sqrt(1 + 1.0 / gamma))) - 2 * math.erf(
                R * (1 - np.sqrt(1 + 1.0 / gamma)))) + 2 * (
                        1 + math.erf(R * (1 - np.sqrt(1 + 1.0 / gamma))) ** 2) * R * R \
            + 2 * (k + math.erf(R * (1 - np.sqrt(1 + 1.0 / gamma))) ** 2) * var_z * var_z - tau ** 2 * (
                        2 * k * gamma + k - math.erf(R * (1 - np.sqrt(1 + 1.0 / gamma))))

            ]