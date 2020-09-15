import sys
import math
import logging
import pickle
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import defaultdict

PAR_flag = False

logging.basicConfig(filename='app.log', filemode='w', format='%(message)s')

def compute_q_noisy_max(counts, noise_eps):
    winner = np.argmax(counts)
    counts_normalized = noise_eps * (counts - counts[winner])
    counts_rest = np.array(
            [counts_normalized[i] for i in range(len(counts)) if i != winner])
    q = 0.0
    for c in counts_rest:
        gap = -c
        q += (2.0 + gap) / (4.0 * math.exp(gap))
    return min(q, 1.0 - (1.0 / len(counts)))

def PAR(e, e_0, gamma):
    return gamma * math.exp(e) + (1 - gamma) * math.exp(e_0)

def logmgf_exact(q, priv_eps, l):
    # if q < min(0.5, ((math.exp(priv_eps) - 1) / (math.exp(2 * priv_eps) - 1))):
    if q < 0.5:
        if PAR_flag:
            t_one = (1 - q) * math.pow((1 - q) / (1 - (PAR(0.08, 0.1, 0.8) * q)), l)
            t_two = q * math.pow(PAR(0.08, 0.1, 0.8), l)
        else:
            t_one = (1 - q) * math.pow((1 - q) / (1 - (math.exp(priv_eps) * q)), l)
            t_two = q * math.exp(priv_eps * l)
        t = t_one + t_two
        try:
            log_t = math.log(t)
        except ValueError:
            logging.error("RAISED ERROR")
            log_t = priv_eps * l
    else:
        log_t = priv_eps * l

    data_dep = log_t
    bun_steinke = 0.5 * priv_eps * priv_eps * l * (l + 1)
    worst = priv_eps * l

    if (data_dep < bun_steinke) and (data_dep < worst):
        logging.warning("1")
        return data_dep
    elif (bun_steinke < worst):
        logging.warning("2")
        return bun_steinke
    else:
        logging.warning("3")
        return worst

def logmgf_from_counts(counts, noise_eps, l):
    q = compute_q_noisy_max(counts, noise_eps)
    return logmgf_exact(q, 2.0 * noise_eps, l)

def calculate_epsilon(votes, noise_eps, noise_delta, l_moments):
    N, NUM_TEACHERS = votes.shape
    counts = np.zeros((N, 10))
    for i in range(N):
        for j in range(NUM_TEACHERS):
            counts[i, votes[i, j]] += 1

    total_log_mgf_nm = np.array([0.0 for _ in l_moments])

    for n in range(N):
        total_log_mgf_nm += np.array(
                [logmgf_from_counts(counts[i], noise_eps, l) for l in l_moments])

    eps_list_nm = (total_log_mgf_nm - math.log(noise_delta)) / l_moments
    return min(eps_list_nm)

def shuffle(vec):
    runs = len(vec)
    n = len(vec[runs - 1])
    return [vec[i][np.random.choice(n, n, replace = False)] for i in range(runs)]

def par_expt(noise_eps, noise_delta, l_moments):
    queries = [100, 500, 1000]
    # queries = [100]
    epsilons_without_par = defaultdict(lambda: defaultdict(list))
    epsilons_with_par = defaultdict(lambda: defaultdict(list))

    n_teachers = [50, 100, 150, 200, 250]
    for t in n_teachers:
        with open("votes_%d.pickle" % t, "rb") as pickled_votes:
            original_votes = pickle.load(pickled_votes)

        np.random.seed(1212)

        for q in queries:
            logging.warning("Teachers,queries === %d,%d" % (t, q))

            original_votes = shuffle(original_votes)
            N_EXPTS = len(original_votes)
            N_EXPTS = 5

            global PAR_flag

            PAR_flag = False
            pool = mp.Pool(processes=4)
            results = ([pool.apply(calculate_epsilon,
                args = (original_votes[run][:q], noise_eps, noise_delta, l_moments)) for run in range(N_EXPTS)])
            epsilons_without_par[t][q] = [np.min(results), np.mean(results), np.median(results), np.max(results)]

            PAR_flag = True
            pool = mp.Pool(processes=4)
            results = ([pool.apply(calculate_epsilon,
                args = (original_votes[run][:q], noise_eps, noise_delta, l_moments)) for run in range(len(original_votes))])
            epsilons_with_par[t][q] = [np.min(results), np.mean(results), np.median(results), np.max(results)]

            print("Finished %d queries" % q)
        print("Finished %d teachers\n" % t)

    print("Min, Mean, Median, Max")
    print("Without PAR")
    for q in queries:
        for t in n_teachers:
            print("%d, %d, " % (q, t), epsilons_without_par[t][q])
    print("With PAR")
    for q in queries:
        for t in n_teachers:
            print("%d, %d, " % (q, t), epsilons_with_par[t][q])

    # for q in queries:
    #     plt.plot(n_teachers, [epsilons_without_par[t][q] for t in n_teachers], label="%d queries (w/o PAR)" % q)
    #     plt.plot(n_teachers, [epsilons_with_par[t][q] for t in n_teachers], label="%d queries (with PAR)" % q)

    # plt.xlabel("#teachers")
    # plt.ylabel("epsilon")
    # plt.legend()
    # plt.yscale('log')
    # plt.show()

if __name__ == "__main__":

    noise_eps = 0.05 # epsilon for each call in noisy_max
    noise_delta = 1e-5
    l_moments = 1 + np.array(range(8))

    par_expt(noise_eps, noise_delta, l_moments)
