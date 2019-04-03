import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


NUM_TRIALS = 2000
BANDIT_PROBS = [0.2, 0.5, 0.75]


class Bandit(object):
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1

    def pull(self):
        return np.random.random() < self.p

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        self.a += x
        self.b += 1 - x


def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for bandit in bandits:
        y = beta.pdf(x, bandit.a, bandit.b)
        plt.plot(x, y, label="real p: %.4f" % bandit.p)
    plt.title("Bandit distributions after %s trials" % trial)
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBS]

    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    for i in range(NUM_TRIALS):
        # take a sample from each bandit
        best_b = None
        max_sample = -1
        all_samples = []  # for debugging
        for b in bandits:
            sample = b.sample()
            all_samples.append("%.4f" % sample)
            if sample > max_sample:
                max_sample = sample
                best_b = b
        if i in sample_points:
            print("Current samples" % all_samples)
            plot(bandits, i)

        # pull the arm for bandit with largest sample
        x = best_b.pull()

        # update the distribution for the bandit whose arm we just pulled
        best_b.update(x)


if __name__ == '__main__':
    experiment()
