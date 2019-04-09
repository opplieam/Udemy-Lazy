import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils import shuffle

from facial.util import getImageData, error_rate, init_weight_and_bias, y2indicator
from facial.ann_tf import HiddenLayer

# image dimensions are expected to be: N x width x height x color
# filter shapes are expected to be: filter width x filter height x input feature maps x output feature maps


def init_filter(shape, pool_size):
    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(
        np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(pool_size))
    )
    return w.astype(np.float32)


class ConvPoolLayer(object):
    def __init__(self, mi, mo, fw=5, fh=5, pool_size=(2, 2)):
        # mi = input feature map size
        # mo = output feature map size
        size = (fw, fh, mi, mo)
        W0 = init_filter(size, pool_size)
        b0 = np.zeros(mo, dtype=np.float32)
        self.W = tf.Variable(W0)
        self.b = tf.Variable(b0)
        self.pool_size = pool_size
        self.params = [self.W, self.b]

    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        p1, p2 = self.pool_size
        pool_out = tf.nn.max_pool(
            conv_out,
            ksize=[1, p1, p2, 1],
            strides=[1, p1, p2, 1],
            padding='SAME'
        )
        return tf.tanh(pool_out)


class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, lr=1e-3, mu=0.99, reg=1e-3, decay=0.99999,
            eps=1e-10, batch_size=30, epochs=3, show_fig=True):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)
        K = len(set(Y))

        # validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)

        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        Yvalid_flat = np.argmax(Yvalid, axis=1)  # for calculating error rate

        # initialize convpool layers
        N, width, height, c = X.shape
        mi = c
        outw = width
        outh = height
        self.convpool_layers = []
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh)
            self.convpool_layers.append(layer)
            outw = outw // 2  # max pool (2, 2)
            outh = outh // 2
            mi = mo

        # initialize mlp layers
        self.hidden_layers = []
        M1 = self.convpool_layer_sizes[-1][0] * outw * outh  # size must be same as output of last convpool layer
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        # logistic regression layer
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.convpool_layers:
            self.params += h.params
        for h in self.hidden_layers:
            self.params += h.params

        tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, K), name='Y')
        act = self.forward(tfX)

        rcost = reg * sum(tf.nn.l2_loss(p) for p in self.params)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=act, labels=tfY)
        ) + rcost
        prediction = self.predict(tfX)
        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)

        n_batches = N // batch_size
        costs = []
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j * batch_size:(j * batch_size + batch_size)]
                    Ybatch = Y[j * batch_size:(j * batch_size + batch_size)]

                    sess.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})
                    if j % 20 == 0:
                        c = sess.run(cost, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        costs.append(c)

                        p = sess.run(prediction, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        e = error_rate(Yvalid_flat, p)
                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", c,
                              "error rate:", e)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for c in self.convpool_layers:
            Z = c.forward(Z)
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])  # Flatten
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        py = self.forward(X)
        return tf.argmax(py, 1)


if __name__ == '__main__':
    X, Y = getImageData()
    # reshape X for tf: N x w x h x c
    X = X.transpose((0, 2, 3, 1))
    print("X.shape:", X.shape)

    model = CNN(
        convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],
        hidden_layer_sizes=[500, 300],
    )
    model.fit(X, Y)