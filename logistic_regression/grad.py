import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)
# center first 50 points at (-2, -2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
# center last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))
# labels: first 50 = 0, last 50 = 1
T = np.array([0] * 50 + [1] * 50)

ones = np.array([[1] * N]).T
Xb = np.concatenate((ones, X), axis=1)

W = np.random.random(D + 1)
z = Xb.dot(W)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


Y = sigmoid(z)


def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


learning_rate = 0.1
w = 0
for i in range(100):
    if i % 10 == 0:
        print(cross_entropy(T, Y))

    w += learning_rate * Xb.T.dot(T - Y)
    Y = sigmoid(Xb.dot(w))

print("Final w:", w)