# coding:utf8
import threading
import Queue

SHARE_Q = Queue.Queue()  # 构造一个不限制大小的的队列
_WORKER_THREAD_NUM = 6  # 设置线程个数
k = 5
res = []


class MyThread(threading.Thread):
    def __init__(self, func):
        super(MyThread, self).__init__()
        self.func = func

    def run(self):
        self.func()


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
    idx = d.argsort()[:k]
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

import time
def worker():
    while not SHARE_Q.empty():
        i = SHARE_Q.get()  # 获得任务
        c = computeFunc(train_X, test_X[i], k)
        print time.ctime(), i, test_y[i], c
        res.append((i, c))


def main():
    threads = []
    num_of_test = test_X.shape[0]
    for task in xrange(num_of_test):  # 向队列中放入任务
        SHARE_Q.put(task)
    for i in xrange(_WORKER_THREAD_NUM):
        thread = MyThread(worker)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    from sklearn.metrics import accuracy_score
    y_pred = [0 for _ in range(test_y.shape[0])]
    for _ in res:
        y_pred[_[0]] = _[1]
    print accuracy_score(test_y, np.array(y_pred))


if __name__ == '__main__':
    main()
