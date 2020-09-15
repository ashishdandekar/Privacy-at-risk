import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

def build_model(x, y):
    return LogisticRegression(solver = 'lbfgs').fit(x, y)

def generate_dataset(x, models, noise_eps, noise = True, local = False):
    # This parts implements noisy max mechanism
    N_classes = len(models[0].classes_)


    y = []
    if local:
        predictions = []
        for model in models:
            temp = model.predict(x)
            select = []
            random = []
            param = (np.e ** noise_eps) / (N_classes - 1 + np.e**noise_eps)
            for i in range(len(temp)):
                if np.random.random() < param:
                    select.append(1)
                    random.append(0)
                else:
                    select.append(0)
                    random.append(np.random.choice(range(N_classes)))
            predictions.append(temp * select + random)

        predictions = np.array(predictions).T
        for datapt in predictions:
            votes = np.zeros(N_classes)
            for vote in datapt: votes[vote] += 1
            y.append(np.argmax(votes))
    else:
        # prediction is samples * model
        predictions = np.array([model.predict(x) for model in models]).T
        for datapt in predictions:
            votes = np.zeros(N_classes)
            for vote in datapt: votes[vote] += 1

            if noise: votes += np.random.laplace(0, 1. / noise_eps, N_classes)

            y.append(np.argmax(votes))

    return (x, y)

def local_experiment(n_teachers, teachers, n_queries, noise_eps):
    np.random.seed(1234)
    plt.clf()
    for t in n_teachers:
        query_analysis_local = defaultdict(list)
        query_analysis_global = defaultdict(list)
        # for i in range(len(teachers[t])):
        for i in range(10):
            for q in n_queries:
                X = test_x[np.random.choice(5000, q, replace = False), :]
                # local
                synth_dataset = generate_dataset(X, teachers[t][i], noise_eps / t, False, True)
                query_analysis_local[q].append(
                        build_model(*synth_dataset).score(validation_x, validation_y))
                # global
                synth_dataset = generate_dataset(X, teachers[t][i], noise_eps, False)
                query_analysis_global[q].append(
                        build_model(*synth_dataset).score(validation_x, validation_y))
            print("Finished iteration: ", i)

        mn = np.array([np.mean(query_analysis_local[q]) for q in n_queries])
        std = np.array([np.std(query_analysis_local[q]) for q in n_queries])
        plt.plot(n_queries, mn, label = "#teachers=%d (local)" % t)
        plt.fill_between(n_queries, mn - std, mn + std, alpha=0.4)

        mn = np.array([np.mean(query_analysis_global[q]) for q in n_queries])
        std = np.array([np.std(query_analysis_global[q]) for q in n_queries])
        plt.plot(n_queries, mn, label = "#teachers=%d (global)" % t)
        plt.fill_between(n_queries, mn - std, mn + std, alpha=0.4)

    plt.xlabel("Number of queries")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    t = time.time()
    test_x, test_y = loadmnist('data/t10k-images-idx3-ubyte',
            'data/t10k-labels-idx1-ubyte')
    print("Time to load datasets: ", time.time() - t)

    validation_x, validation_y = test_x[5000:], test_y[5000:]

    teachers = {}
    # n_teachers = [50, 100, 150, 200, 250]
    # n_teachers = [50, 100, 150]
    n_teachers = [50, 100]
    for n in n_teachers:
        teachers[n] = pickle.load(open("teachers_%d.pickle" % n, "rb"))

    n_queries = np.arange(0, 1000, 50) + 50
    noise_eps = 0.05

    local_experiment(n_teachers, teachers, n_queries, noise_eps)
    sys.exit(0)

    np.random.seed(1234)

    # Query analysis
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for t in n_teachers:
        query_analysis = defaultdict(list)
        # for i in range(len(teachers[t])):
        for i in range(10):
            for q in n_queries:
                X = test_x[np.random.choice(5000, q, replace = False), :]
                # local
                synth_dataset = generate_dataset(X, teachers[t][i], noise_eps / t, False, True)
                # global
                # synth_dataset = generate_dataset(X, teachers[t][i], noise_eps, False, True)
                # synth_dataset = generate_dataset(X, teachers[t][i], noise_eps)
                query_analysis[q].append(
                        build_model(*synth_dataset).score(validation_x, validation_y))
            print("Finished iteration: ", i)

        # ax1.boxplot([query_analysis[q] for q in n_queries])
        mn = np.array([np.mean(query_analysis[q]) for q in n_queries])
        std = np.array([np.std(query_analysis[q]) for q in n_queries])
        ax1.plot(n_queries, mn, label = "#teachers=%d" % t)
        ax1.fill_between(n_queries, mn - std, mn + std, alpha=0.4)

    # ax1.set_xticks(np.array(range(len(n_queries))) + 1, map(str, n_queries))
    ax1.set_xlabel("Number of queries")
    ax1.set_ylabel("Accuracy")

    with open("data_dependent_eps.pickle", "rb") as eps:
        epsilons = pickle.load(eps)

    ax2 = ax1.twinx()
    ax2.plot(n_queries, [epsilons[250][i] for i in n_queries], color = "black", linestyle = "dotted")
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ax2.set_ylabel("Data dependent $\epsilon$")

    fig.legend()
    plt.show()
    # plt.savefig("student_query_accuracy.png", bbox_inches = "tight", figsize=(15, 15))

    # teacher analysis
    # We fix number of 
    # teacher_analysis = defaultdict(list)
    # for i in range(N_EXPTS):
    #     X = test_x[np.random.choice(5000, 500, replace = False), :]
    #     for t in n_teachers:
    #         synth_dataset = generate_dataset(X, teachers[t], noise_eps)
    #         teacher_analysis[t].append(
    #                 build_model(*synth_dataset).score(validation_x, validation_y))
    #     print("Finished iteration: ", i)
    # plt.clf()
    # plt.boxplot([teacher_analysis[q] for q in n_teachers])
    # plt.xticks(np.array(range(len(teachers))) + 1, map(str, n_teachers))
    # plt.xlabel("Number of teachers")
    # plt.ylabel("Accuracy")
    # plt.title("With 500 queries")
    # plt.savefig("student_teacher_accuracy.png", bbox_inches = "tight")
