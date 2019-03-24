import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('./data/data_2d.csv', header=None)
X = df[[0, 1]].values
Y = df[2].values
bias = np.ones((100, 1))
X = np.append(bias, X, axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
y_hat = np.dot(X, w)

d1 = Y - y_hat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("R2", r2)