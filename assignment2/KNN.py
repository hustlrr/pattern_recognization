# coding=utf-8

import numpy as np
from scipy.spatial.distance import cosine

train_X = np.load(r'../MNIST/train_X.npy') / 255.0
train_y = np.load(r'../MNIST/train_y.npy')

test_X = np.load(r'../MNIST/test_X.npy') / 255.0
test_y = np.load(r'../MNIST/test_y.npy')


def computeFunc(X, x, k):
    n = X.shape[0]
    d = np.array([0 for _ in range(n)], dtype=float)
    for j in range(n):
        d[j] = cosine(x, X[j])
    idx = d.argpartition(kth=k)[:k]
    category = {}
    for i in idx:
        try:
            category[train_y[i]] += 1
        except KeyError:
            category[train_y[i]] = 1
    maxcnt, maxcategory = 0, 0
    for k, v in category.items():
        if v > maxcnt:
            maxcnt = v
            maxcategory = k
    return maxcategory


def myaccuracyScore(y_ture, y_pred):
    return sum(map(lambda _: _[0] == _[1], zip(y_ture, y_pred))) * 1.0 / len(y_ture)


import time

for k in [1, 5, 10, 15]:
    pred_y = []
    num_of_test = test_X.shape[0]
    for i in range(num_of_test):
        x = test_X[i]
        y = computeFunc(train_X, x, k)
        pred_y.append(y)
        print time.ctime(), i, test_y[i], y
    print 'k = ', k, 'accuracy=', myaccuracyScore(pred_y, test_y)