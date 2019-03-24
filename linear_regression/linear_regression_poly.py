import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/data_poly.csv', header=None, names=['x', 'y'])
df['b'] = 1
df['poly'] = df['x'] * df['x']

X = df[['b', 'x', 'poly']].values
Y = df['y'].values

plt.scatter(X[:, 1], Y)
plt.show()

# calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
y_hat = np.dot(X, w)

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(y_hat))
plt.show()