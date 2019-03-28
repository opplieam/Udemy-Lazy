from supervised.util import get_data
from scipy.stats import multivariate_normal as mvn
import numpy as np


class NaiveBayes(object):

    def fit(self, X, Y, smoothing=1e-2):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data(10000)
    N_train = len(Y) // 2
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    model = NaiveBayes()
    model.fit(X_train, Y_train)

    print("Train accuracy:", model.score(X_train, Y_train))
    print("Test accuracy:", model.score(X_test, Y_test))
