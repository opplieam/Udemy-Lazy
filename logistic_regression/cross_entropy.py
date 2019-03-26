import numpy as np
import matplotlib.pyplot as plt

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


print(cross_entropy(T, Y))

# Closed form
W = np.array([0, 4, 4])
z = Xb.dot(W)
Y = sigmoid(z)
print(cross_entropy(T, Y))

# y = -x
plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
