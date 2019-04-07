import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z


def classification_rate(Y, P):
    return np.mean(Y == P)


def derivative_w2(Z, T, Y):
    return Z.T.dot(T - Y)


def derivative_w1(X, Z, T, Y, W2):
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    return X.T.dot(dZ)


def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)


def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def main():
    # create the data
    Nclass = 500
    D = 2  # dimensionality of input
    M = 3  # hidden layer size
    K = 3  # number of classes

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)
    # turn Y into an indicator matrix for training
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    # let's see what it looks like
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()

    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 1e-3
    costs = []
    for epoch in range(1000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("cost:", c, "classification_rate:", r)
            costs.append(c)

        # this is gradient ASCENT, not DESCENT
        # be comfortable with both!
        # oldW2 = W2.copy()
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()