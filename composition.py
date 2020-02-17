from mpmath import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from functools import reduce
import math

def regularizedHyper(a, b, z):
    '''
    This is a helper function.
    '''
    return hyper(a, b, z) / reduce(lambda x, y: x * y, map(gamma, b))

def integral(x, n):
    '''
    This is a helper function.
    '''
    term1 = (np.pi ** 0.5) * (4 ** n) * x *regularizedHyper([0.5], [1.5-n, 1.5], (x**2)/4)
    term2 = 2 * (x ** (2 *n)) * gamma(n) * regularizedHyper([n], [n+0.5, n+1], (x**2)/4)
    return term1 - term2

def bound(e1, e2, n):
    '''
    This function computes $\gamma_1$ for the specified values of
    epsilon_1 and epsilon_2. $n$ denotes the dimension of the output
    of the Laplace mechnaism. For scalar output, choose $n = 1$.
    '''
    return integral(e1, n) / integral(e2, n)

def KL_bound(eps):
    return eps * (math.e**eps - 1)

def composition(eps, k, bound = "hoeffding", delta = 0.0001):
    if bound == "hoeffding":
        term1 = ((1. / (2*k)) * math.log(1 / (1 - delta)))**0.5 * eps
    elif bound == "azuma":
        term1 = ((2 * k * math.log(1 / delta))**0.5) * eps
    elif bound == "bernstein":
        term1 = math.log(1./delta) * eps * (1/3.)
        term1 *= (1 + math.sqrt(1 + (1.5 * k)/math.log(1./delta)))
    term2 = k * KL_bound(eps)
    return term1 + term2

def composition_par(eps0, eps, k, delta = 0.0001):
    ee = lambda i: i * math.exp(i)
    gamma = bound(eps, eps0, 1)
    term1 = ((1. / (2*k)) * math.log(1 / (1 - delta)))**0.5 * (eps0 - eps)
    term2 = k * ((gamma * KL_bound(eps)) + ((1 - gamma) * KL_bound(eps0)))

    return term1 + term2

def best_eps(e, e_0):
    term1 = (1./ e) - (1. / e_0)
    term2 = math.log(1 - ((1 - math.e**e)/(e**2)))
    return term1 - term2

if __name__ == "__main__":
    ks = range(1, 300)

    eps0s = [0.05, 0.1, 0.5, 1.0]
    opt_eps = []
    for eps0 in eps0s:
        opt_eps.append(optimize.newton(best_eps, 0.002, args = (eps0,)))

    print("For 1000 teachers with 0.1-DP mechanism")
    print("Composition with PAR:", composition_par(0.1, opt_eps[1], 1000, delta = 1e-6))
    print("Composition with Azuma:", composition(0.1, 1000, "azuma", delta = 1e-6))

    # plt.clf()
    # for i, eps_0 in enumerate(eps0s):
    #     plt.plot(ks, [composition_par(eps_0, opt_eps[i], k) for k in ks], label = "%.1f" % eps_0) 
    # plt.legend()
    # plt.xlabel("k")
    # plt.ylabel("$\epsilon'$")
    # plt.show()

    for i, eps_0 in enumerate(eps0s):
        plt.clf()
        plt.plot(ks, [composition(eps_0, k, "azuma") for k in ks], label = "advanced_azuma")
        # plt.plot(ks, [composition(eps_0, k, "bernstein") for k in ks], label = "advanced_bernstein")
        # plt.plot(ks, [composition(eps_0, k, "hoeffding") for k in ks], label = "advanced_hoeffding")
        plt.plot(ks, [eps_0 * k for k in ks], label = "basic")
        plt.plot(ks, [composition_par(eps_0, opt_eps[i], k) for k in ks], label = "par")
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("$\epsilon$ for $\delta = 0.0001$")
        plt.title("Advanced composition for $\epsilon_0 = %0.2f$" % eps_0)
        plt.savefig("advance_%d.png" % i, bbox_inches = "tight")

    # plt.clf()
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(eps0s, opt_eps, "b-", marker='o')
    # ax2.plot(eps0s, [bound(j, i, 1) for i, j in zip(eps0s, opt_eps)], "r-", marker='+')

    # ax1.set_xlabel("$\epsilon_0$")
    # ax1.set_ylabel("Optimial privacy at risk $\epsilon_{opt}$", color='b')
    # ax2.set_ylabel("Optimial privacy at risk $\gamma_{opt}$", color='r')
    # plt.show()
