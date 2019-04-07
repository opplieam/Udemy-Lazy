import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv('./data/ecommerce_data.csv')
    data = df.values
    np.random.shuffle(data)

    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)

    # one-hot encode the categorical data
    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]  # non-categorical

    # one-hot
    for n in range(N):
        t = int(X[n, D - 1])
        X2[n, t + D - 1] = 1

    X = X2

    # split train and test
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]

    # normalize columns 1 and 2
    for i in (1, 2):
        m = Xtrain[:, i].mean()
        s = Xtrain[:, i].std()
        Xtrain[:, i] = (Xtrain[:, i] - m) / s
        Xtest[:, i] = (Xtest[:, i] - m) / s

    return Xtrain, Ytrain, Xtest, Ytest

