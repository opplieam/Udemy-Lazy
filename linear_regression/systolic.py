# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('./data/mlr02.xls')
df['ones'] = 1
X = df[['X2', 'X3', 'ones']]
Y = df['X1']

plt.scatter(X['X2'], Y)
plt.show()

plt.scatter(X['X3'], Y)
plt.show()

X2_only = df[['X2', 'ones']]
X3_only = df[['X3', 'ones']]
Exclude_noise = df[['X2', 'X3']]


def get_r2(x, y):
    w = np.linalg.solve(x.T.dot(x), x.T.dot(y))
    y_hat = x.dot(w)

    d1 = y - y_hat
    d2 = y - y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2


print("R2 for X2 only", get_r2(X2_only, Y))
print("R2 for X3 only", get_r2(X3_only, Y))
print("R2 for both", get_r2(X, Y))
print("R2 for both exclude noise", get_r2(Exclude_noise, Y))
