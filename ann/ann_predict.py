import numpy as np
from ann.ecom_process import get_data


X, Y, _, _ = get_data()

# randomly initialize weights
M = 5  # hidden unit
N = X.shape[0]  # samples size
D = X.shape[1]  # no. features
K = len(set(Y))  # no. target

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)


def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = X.dot(W1) + b1
    Z = np.tanh(Z)
    return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
predictions = np.argmax(P_Y_given_X, axis=1)


def classification_rate(Y, P):
    return np.mean(Y == P)


print("Score", classification_rate(Y, predictions))