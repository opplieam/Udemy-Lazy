# K-Nearest Neighbors classifier on MNIST data.
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sortedcontainers import SortedList
from supervised.util import get_data


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):  # test points
            sl = SortedList()  # stores (distance, class) tuples
            for j, xt in enumerate(self.X):  # training points
                diff = x - xt
                d = diff.dot(diff)  # square distance
                if len(sl) < self.k:
                    # don't need to check, just add
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:  # check from the last item
                        del sl[-1]
                        sl.add((d, self.y[j]))
            # vote
            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1

            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v

            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        p = self.predict(X)
        return np.mean(p == Y)


if __name__ == '__main__':
    X, Y = get_data(2000)
    N_train = 1000
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    train_scores = []
    test_scores = []
    ks = (1, 2, 3, 4, 5)
    for k in ks:
        print("K:", k)
        knn = KNN(k)

        t0 = datetime.now()
        knn.fit(X_train, Y_train)
        print("Training time:", datetime.now() - t0)

        t0 = datetime.now()
        train_score = knn.score(X_train, Y_train)
        train_scores.append(train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0),
              "Train size:", len(Y_train))

        t0 = datetime.now()
        test_score = knn.score(X_test, Y_test)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0),
              "Test size:", len(Y_test))

    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()
