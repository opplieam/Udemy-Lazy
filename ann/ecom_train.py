import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from ann.ecom_process import get_data

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

Xtrain, Ytrain, Xtest, Ytest = get_data()
D = Xtrain.shape[1]
K = len(set(Ytrain) | set(Ytest))
M = 5 # num hidden units

# convert to indicator
Ytrain_ind = y2indicator(Ytrain, K)
Ytest_ind = y2indicator(Ytest, K)

# randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)


# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z


def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)


# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)


def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))

# train loop
train_costs = []
test_costs = []
learning_rate = 0.001
epochs = 10000
for i in range(epochs):
    pY_train, Z_train = forward(Xtrain, W1, b1, W2, b2)
    pY_test, Z_test = forward(Xtest, W1, b1, W2, b2)

    ctrain = cross_entropy(Ytrain_ind, pY_train)
    ctest = cross_entropy(Ytest_ind, pY_test)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent
    W2 -= learning_rate * Z_train.T.dot(pY_train - Ytrain_ind)
    b2 -= learning_rate * (pY_train - Ytrain_ind).sum(axis=0)
    dZ = (pY_train - Ytrain_ind).dot(W2.T) * (1 - Z_train * Z_train)
    W1 -= learning_rate * Xtrain.T.dot(dZ)
    b1 -= learning_rate * dZ.sum(axis=0)
    if i % 1000 == 0:
        print(i, ctrain, ctest)

print("Final train classification_rate:", classification_rate(Ytrain, predict(pY_train)))
print("Final test classification_rate:", classification_rate(Ytest, predict(pY_test)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()