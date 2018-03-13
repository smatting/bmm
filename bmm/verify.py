import numpy as np
from scipy.stats import bernoulli
from bmm import BMM


def gen_samples(n, q, q_h, seed=3):
    np.random.seed(seed)
    j = np.random.choice(len(q_h), size=n, p=q_h)
    p = q[:, j].T
    samples = bernoulli.rvs(p)
    return samples


def main():
    n = 5000
    q = np.array([[0.3, 0.9],
                  [0.8, 0.1],
                  [0.2, 0.6]])
    q_h = np.array([0.2, 0.8])
    print(q)
    print(q_h)
    v = gen_samples(n, q, q_h)

    bmm = BMM(n_comp=2, n_runs=10, thresh=1e-5)
    bmm.fit(v)
    print(bmm.q_)
    print(bmm.q_h_)


if __name__ == '__main__':
    main()
