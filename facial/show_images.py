import numpy as np
import matplotlib.pyplot as plt

from facial.util import getData

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

if __name__ == '__main__':
    X, Y = getData(balance_ones=False)
    while True:
        for i in range(7):
            x, y = X[Y == i], Y[Y == i]
            N = len(y)
            j = np.random.choice(N)
            plt.imshow(x[j].reshape(48, 48), cmap='gray')
            plt.title(label_map[y[j]])
            plt.show()
        prompt = input('Quit? Enter Y: \n')
        if prompt == 'Y':
            break
