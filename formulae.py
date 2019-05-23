import numpy as np
from mpmath import *

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

def calculate_gamma(e1, e2, rho, delta, alpha, n):
	'''
		This function computes $\gamma_3$.
		rho: tolerance parameter
		delta: sampled sensitivity
		alpha: accuracy parameter
	'''
	eta = 1. + 2 * (rho / delta)
	return (integral(e1, n) / integral(e2*eta, n)) * alpha

def calculate_nsamples(rho, alpha):
	'''
		This function returns the number samples required
		to meet the accuracy requirement $\alpha$ and
		the tolerance parameter $\rho$
	'''
	return (1. / (2 * (rho**2))) * np.log(2. / (1 - alpha))
