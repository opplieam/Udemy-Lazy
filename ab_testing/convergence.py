import numpy as np
import matplotlib.pyplot as plt
from ab_testing.bayesian_bandit import Bandit


def run_experiment(p1, p2, p3, N):
    bandits = [Bandit(p1), Bandit(p2), Bandit(p3)]

    data = np.empty(N)

    for i in range(N):
        # Thompson sampling
        j = np.argmax([b.sample() for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        # For the plot
        data[i] = x

    cumulative_avg_ctr = np.cumsum(data) / (np.arange(N) + 1)

    # plot moving avg ctr
    plt.plot(cumulative_avg_ctr)
    plt.plot(np.ones(N) * p1)
    plt.plot(np.ones(N) * p2)
    plt.plot(np.ones(N) * p3)
    plt.ylim((0, 1))
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    run_experiment(0.2, 0.25, 0.3, 10000)
