import time
import pickle
import numpy as np
import multiprocessing as mp
from sklearn import svm
from sklearn.linear_model import LogisticRegression

N_TEACHERS = 50
N_RUNS = 30

def shuffle_data(x, y):
    n = len(y)
    idx = np.random.choice(n, n, replace = False)
    return x[idx], y[idx]

def split_data(x, y):
    return zip(np.split(x, N_TEACHERS), np.split(y, N_TEACHERS))

def build_logistic(x, y):
    return LogisticRegression(solver = 'lbfgs').fit(x, y)

if __name__ == "__main__":
    t = time.time()
    train_x, train_y = loadmnist('data/train-images-idx3-ubyte',
            'data/train-labels-idx1-ubyte')
    test_x, test_y = loadmnist('data/t10k-images-idx3-ubyte',
            'data/t10k-labels-idx1-ubyte')
    print("Time to load datasets: ", time.time() - t)

    # t = time.time()
    # model = build_svm(train_x[:5000], train_y[:5000])
    # print("Time for training: ", time.time() - t)

    # print("Accuracy: ", model.score(test_x, test_y))

    # np.random.seed(1234)

    models = []
    votes = []
    for r in range(N_RUNS):
        train_x, train_y = shuffle_data(train_x, train_y)
        datasets = split_data(train_x, train_y)

        pool = mp.Pool(processes=6)
        t = time.time()
        models.append([pool.apply(build_logistic, args=(x, y)) for x, y in datasets])
        print("Time for training: ", time.time() - t)

        # pool = mp.Pool(processes=6)
        # accuracy = [pool.apply(model.score, args=(test_x, test_y)) for model in models[-1]]
        # print(accuracy)

        pool = mp.Pool(processes=6)
        votes.append(np.array([pool.apply(model.predict, args=(test_x,)) for model in models[-1]]).T)
        print("Finished run: %d", r)

    with open("teachers_%d.pickle" % N_TEACHERS, 
            "wb") as teachers, open("votes_%d.pickle" % N_TEACHERS, "wb") as fvotes:
        pickle.dump(models, teachers)
        pickle.dump(votes, fvotes)
