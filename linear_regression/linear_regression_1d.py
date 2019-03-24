import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./data/data_1d.csv', header=None)
x = df[0].values
y = df[1].values

plt.scatter(x, y)
plt.show()

# apply the equations to calculate a and b
denominator = x.dot(x) - x.mean() * x.sum()

a = (x.dot(y) - y.mean() * x.sum()) / denominator
b = (y.mean() * x.dot(x) - x.mean() * x.dot(y)) / denominator

# calculate predicted Y
y_hat = a * x + b

plt.scatter(x, y)
plt.plot(x, y_hat)
plt.show()