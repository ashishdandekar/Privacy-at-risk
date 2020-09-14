import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpmath import *
import functools
from sklearn.svm import SVC
from collections import defaultdict
from multiprocessing import Process
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression

def regularizedHyper(a, b, z):
    '''
            This is a helper function.
    '''
    return hyper(a, b, z) / functools.reduce(lambda x, y: x * y, map(gamma, b))

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

def L1(x, y):
    return sum(abs(x - y))

def ridge_regression(data1, data2, output_var, lamb):
    x1, y1 = np.array(data1.loc[:, data1.columns != output_var]), np.array(data1[output_var])
    x2, y2 = np.array(data2.loc[:, data2.columns != output_var]), np.array(data2[output_var])

    x1, x2 = normalize(x1), normalize(x2)

    model1 = Ridge(alpha = lamb * 2)
    model1.fit(x1, y1)
    model2 = Ridge(alpha = lamb * 2)
    model2.fit(x2, y2)

    return L1(model1.coef_, model2.coef_)

def sensitivity_ridge(data, runs):
    output_var = 'INCWAGE'

    lamb = 0.01
    train, test = train_test_split(data, test_size = 0.2)
    # global_delta = (12 * (1./(lamb**0.5)) + 8) / (len(train) * lamb)
    global_delta =  ((1./(lamb ** 0.5)) + 1) * (5. / (len(train) * lamb))

    xtrain, ytrain = np.array(train.loc[:, train.columns != output_var]), np.array(train[output_var])
    xtest, ytest = np.array(test.loc[:, test.columns != output_var]), np.array(test[output_var])

    xtrain, xtest = normalize(xtrain), normalize(xtest)
    model = Ridge(alpha = 2 * lamb)
    model.fit(xtrain, ytrain)
    orig_coef = model.coef_
    non_private_accuracy = (sum((model.predict(xtest) - ytest) ** 2) / len(test)) ** 0.5
    print("Ridge fit: ", non_private_accuracy)
    # print("Ridge fit: ", (sum((model.predict(xtest) - ytest) ** 2) / len(test)) ** 0.5)

    confidence = []
    utility_global = defaultdict(list)
    epsilons = np.linspace(0.1, 0.99, 10)
    for e in epsilons:
        confidence.append(bound(e, 1.0, 1))
        for _ in range(runs):
            model.coef_ = orig_coef + np.random.laplace(0, (global_delta / e), orig_coef.shape)
            utility_global[e].append((sum((model.predict(xtest) - ytest) ** 2) / len(xtest)) ** 0.5)
    result = {"confidence": confidence, "utility_global": utility_global, "utility": non_private_accuracy}
    pickle.dump(result, open("new_ridge_utility.pickle", "wb"))

def make_plot():
    result = pickle.load(open("new_ridge_utility.pickle", "rb"))
    utility_global = result['utility_global']

    epsilons = np.linspace(0.1, 0.99, 10)
    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel('Privacy at Risk level ($\epsilon$)')
    ax1.set_ylabel('Confidence level ($\gamma_1$)', color = color)
    ax1.plot(epsilons, [bound(e, 1.0, 1) for e in epsilons], "b--", label = "$\epsilon_0 = %.1f$" % 1.0)
    ax1.plot(epsilons, [bound(e, 0.8, 1) for e in epsilons], "b-+", label = "$\epsilon_0 = %.1f$" % 0.8)
    ax1.plot(epsilons, [bound(e, 0.6, 1) for e in epsilons], "b-*", label = "$\epsilon_0 = %.1f$" %10.6)
    ax1.tick_params(axis = 'y', labelcolor = color)
    ax1.legend()

    ax2 = ax1.twinx()

    mean_global = np.array([np.mean(utility_global[e]) for e in epsilons])
    std_global = np.array([np.std(utility_global[e]) for e in epsilons])

    color = "tab:red"
    ax2.set_ylabel("RMSE", color = color)
    ax2.plot(epsilons, [result['utility']] * len(epsilons), label = "Non private ridge regression", color = "red")
    ax2.plot(epsilons, mean_global, label = "Differentially private ridge regression", linestyle = '--', color = "red")
    ax2.fill_between(epsilons, mean_global - std_global, mean_global + std_global, color = "red", alpha = 0.4)
    ax2.tick_params(axis = 'y', labelcolor = color)
    ax2.legend()

    fig.tight_layout()
    plt.savefig("utility_ridge.png", bbox_inches = "tight")
#     plt.show()

if __name__ == "__main__":
    data_orig = pd.read_csv("census_dataset.csv")
    del data_orig['ID']

    data = data_orig[data_orig['INCWAGE'] != 0]
    print("Dataset Size:", len(data))

    regression_data = data.copy()
    Y = data['INCWAGE']
    Y = (Y - min(Y)) / (max(Y) - min(Y))
    regression_data['INCWAGE'] = Y

    sensitivity_ridge(regression_data.sample(20000), 50)
    make_plot()
