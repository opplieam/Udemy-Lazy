import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from logistic_regression.process_ecomerce import get_binary_data


X, Y = get_binary_data()
X, Y = shuffle(X, Y)
X_train = X[:-100]
Y_train = Y[:-100]
X_test = X[-100:]
Y_test = Y[-100:]

D = X.shape[1]
W = np.random.randn(D)
b = 0


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


def classification_rate(Y, P):
    return np.mean(Y == P)


def cross_entropy(T, pY):
    return -np.mean(T * np.log(pY) + (1 - T) * np.log(1 - pY))


train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    py_train = forward(X_train, W, b)
    py_test = forward(X_test, W, b)

    c_train = cross_entropy(Y_train, py_train)
    c_test = cross_entropy(Y_test, py_test)

    train_costs.append(c_train)
    test_costs.append(c_test)

    W -= learning_rate * X_train.T.dot(py_train - Y_train)
    b -= learning_rate * (py_train - Y_train).sum()
    if i % 1000 == 0:
        print(i, c_train, c_test)


print("Final train classification rate:", classification_rate(Y_train, np.round(py_train)))
print("Final test classification rate:", classification_rate(Y_test, np.round(py_test)))

legend1 = plt.plot(train_costs, label='train cost')
legend2 = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
