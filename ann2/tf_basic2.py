import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ann2.util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)


def main():
    X_train, X_test, y_train, y_test = get_normalized_data()

    max_iter = 15
    print_period = 30

    y_train_ind = y2indicator(y_train)
    y_test_ind = y2indicator(y_test)

    N, D = X_train.shape
    batch_size = 500
    n_batches = N // batch_size

    M1 = 300
    M2 = 100
    K = 10
    W1_init = np.random.randn(D, M1) / np.sqrt(D)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)

    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Y_out = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_out, labels=T)
    )
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    predict_op = tf.argmax(Y_out, 1)

    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = X_train[j * batch_size:(j * batch_size + batch_size), ]
                Ybatch = y_train_ind[j * batch_size:(j * batch_size + batch_size), ]

                sess.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = sess.run(cost,
                                            feed_dict={X: X_test, T: y_test_ind})
                    prediction = sess.run(predict_op, feed_dict={X: X_test})
                    err = error_rate(prediction, y_test)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (
                    i, j, test_cost, err))
                    costs.append(test_cost)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()
