# coding=utf-8

import numpy as np

sigmoid = np.vectorize(lambda _: (1.0 / (1.0 + np.exp(-1.0 * _))))


def pred(W1, W2, X, y):
    l0 = X  # 输入层的输出
    net1 = np.dot(l0, W1)
    l1 = np.tanh(net1)  # 隐藏层输出
    net2 = np.dot(l1, W2)
    l2 = sigmoid(net2)  # 输出层的输出
    predy = np.argmax(l2, axis=1)
    jw = np.sum(np.square(y - l2), dtype=float) * 0.5  # 准则函数
    return predy, jw

def backpropogation(X, y, d=3, nh=8, c=3, eta=1, MAXITER=15000):
    # initial
    np.random.seed(10)
    W1 = np.random.random((d, nh)) - 1  # 输入层到隐藏层的权重
    W2 = np.random.random((nh, c)) - 1  # 隐藏层到输出层的权重
    np.random.seed(None)
    jws = []

    for i in range(MAXITER):
        idx = np.random.choice(X.shape[0], 1)
        x_ = X[idx].reshape(1, d)
        y_ = y[idx]
        # forward
        l0 = x_  # 输入层的输出
        net1 = np.dot(l0, W1)
        l1 = np.tanh(net1)  # 隐藏层输出
        net2 = np.dot(l1, W2)
        l2 = sigmoid(net2)  # 输出层的输出
        # jw = np.sum(np.square(y - l2), dtype=float) * 0.5  # 准则函数
        # 判断是否终止
        predy, jw = pred(W1, W2, X, y)
        jws.append(jw)
        if (predy == np.argmax(y, axis=1)).all():
            return predy, i
        # backpropogation
        delta2 = -(y_ - l2) * l2 * (1 - l2)
        deltaW2 = np.dot(l1.T, delta2)
        delta1 = np.dot(delta2, W2.T) * (1 - np.square(l1))
        deltaW1 = np.dot(l0.T, delta1)
        # update
        W2 = W2 - eta * deltaW2
        W1 = W1 - eta * deltaW1

    predy, jw = pred(W1, W2, X, y)
    return predy, MAXITER, jws
