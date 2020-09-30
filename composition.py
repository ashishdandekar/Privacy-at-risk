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
