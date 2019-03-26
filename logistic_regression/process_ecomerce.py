import numpy as np
import pandas as pd


# 0 is_mobile,
# 1 n_products_viewed,
# 2 visit_duration,
# 3 is_returning_visitor,
# 4 time_of_day,
# 5 user_action
def get_data():
    df = pd.read_csv('./data/ecommerce_data.csv')
    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]

    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, 0:(D - 1)] = X[:, 0:(D - 1)]

    # One hot encoding for time_of_day
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t + D-1] = 1

    # Alternative
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    # X2[:, -4:] = Z
    assert np.abs(X2[:, -4:] - Z).sum() < 10e-10

    return X2, Y


def get_binary_data():
    # For logistic we want user_action for only 0 or 1
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2
