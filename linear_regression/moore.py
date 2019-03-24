import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


non_decimal = re.compile(r'[^\d]+')

df = pd.read_csv('./data/moore.csv', delimiter='\t', header=None)
x = df[2].apply(
    lambda v: non_decimal.sub('', v.split('[')[0])
)
y = df[1].apply(
    lambda v: non_decimal.sub('', v.split('[')[0])
)

x = x.values.astype(np.int)
y = y.values.astype(np.int)

plt.scatter(x, y)
plt.show()

# Transform y to be more like linear
y = np.log(y)
plt.scatter(x, y)
plt.show()

denominator = x.dot(x) - x.mean() * x.sum()

a = (x.dot(y) - y.mean() * x.sum()) / denominator
b = (y.mean() * x.dot(x) - x.mean() * x.dot(y)) / denominator

y_hat = a * x + b

plt.scatter(x, y)
plt.plot(x, y_hat)
plt.show()

d1 = y - y_hat
d2 = y - y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("a:", a, "b", b)
print('R2', r2)

# log(tc) = a * year + b
# tc = exp(b) * exp(a * year)
# Time to double
# 2*tc = 2 * exp(b) * exp(a * year)
#      = exp(ln(2)) * exp(b) * exp(a * year)
#      = exp(b) * exp(a * year * ln(2))
# exp(b) * exp(a * year2) = exp(b) * exp(a * year1 + ln2)
# a * year2 = a * year1 + ln2
# year2 = year1 + ln2/a
print('time to double:', np.log(2) / a, 'years')
