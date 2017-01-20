# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def batchperception(X, y):
    '''
    :param X: np.array
    :param y: np.array
    :return:
    '''
    X = np.c_[np.ones(X.shape[0]), X]
    d = X.shape[1]
    a = np.zeros(shape=(d,), dtype=float)
    X[y < 0] = -X[y < 0]
    cnt = 0  # cnt is the number of iterations
    while True:
        errors = [x for x in X if np.dot(a.T, x.T) <= 0]
        if not len(errors):
            print a
            break
        a = a + 1.0 * (reduce(lambda x, y: x + y, errors).T)
        cnt += 1
        print cnt
    print 'finish'
    return a

def jr(a, x, b):
    return 0.5 * np.square(np.dot(x, a) - b) / (np.sum(np.square(x)))

def batchRelaxtion(X, y, name, eta=2, b=0.1, MAXCNT=1000):
    f1 = plt.scatter(X[y > 0][:, 0], X[y > 0][:, 1], color='r')
    f2 = plt.scatter(X[y < 0][:, 0], X[y < 0][:, 1], color='b')
    plt.legend([f1, f2], name, loc=4)
    plt.title(r' batch relaxation with margin, b=%.1f'%b)

    X = np.c_[np.ones(X.shape[0]), X]
    X[y < 0] = -X[y < 0]
    a = np.zeros(shape=(X.shape[1],), dtype=float)
    ja = []
    for cnt in range(MAXCNT):
        errors = [x for x in X if np.dot(x, a) <= b]
        if not len(errors):
            print a, cnt
            break
        ja.append([cnt, np.sum([jr(a, x, b) for x in errors])])
        for x in errors:
            a = a - eta * ((np.dot(x, a) - b) / (np.sum(np.square(x)))) * x
        print cnt
    x1 = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.1)
    x2 = -a[0] / a[2] - a[1] / a[2] * x1
    plt.plot(x1, x2, color='black')
    plt.show()
    ja = np.array(ja)
    return a, ja


def singleRelaxtion(X, y, name, eta=2, b=0.1, MAXCNT=1000):
    f1 = plt.scatter(X[y > 0][:, 0], X[y > 0][:, 1], color='r')
    f2 = plt.scatter(X[y < 0][:, 0], X[y < 0][:, 1], color='b')
    plt.legend([f1, f2], name, loc=4)
    plt.title(r' batch relaxation with margin, b=%.1f' % b)

    X = np.c_[np.ones(X.shape[0]), X]
    X[y < 0] = -X[y < 0]
    ja = []
    a = np.zeros(shape=(X.shape[1],), dtype=float)
    for cnt in range(MAXCNT):
        errors = [x for x in X if np.dot(x, a) <= b]
        if not len(errors):
            print a, cnt
            break
        ja.append([cnt, np.sum([jr(a, x, b) for x in errors])])
        x = X[cnt % X.shape[0]]
        if np.dot(x, a) <= b:
            a = a - eta * ((np.dot(x, a) - b) / (np.sum(np.square(x)))) * x
    x1 = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.1)
    x2 = -a[0] / a[2] - a[1] / a[2] * x1
    plt.plot(x1, x2, color='black')
    plt.show()
    ja = np.array(ja)
    return a, ja


def js(X, a, b):
    return np.sum(np.square(np.dot(X, a) - b))

def HoKashyap(X, y, name, bmin=0.1, KMAX=1000, eta=0.5):
    f1 = plt.scatter(X[y > 0][:, 0], X[y > 0][:, 1], color='r')
    f2 = plt.scatter(X[y < 0][:, 0], X[y < 0][:, 1], color='b')
    plt.legend([f1, f2], name, loc=4)
    plt.title(r'Ho-Kashyap algorithm')

    X = np.c_[np.ones(X.shape[0]), X]
    X[y < 0] = -X[y < 0]
    a = np.zeros(shape=(X.shape[1]))
    b = np.ones(shape=(X.shape[0]))
    for cnt in range(KMAX):
        e = np.dot(X, a) - b
        e_plus = 0.5 * (e + np.abs(e))
        b = b + 2 * eta * e_plus
        a = np.dot(np.linalg.pinv(X), b)
        if (np.abs(e) <= bmin).all():
            print 'found, iter = %d, js = %.3f' % (cnt, js(X, a, b))
            break
        if ((e < 0).all() or not e_plus.any()) and cnt:
            print 'no seperable, iter = %d, js = %.3f' % (cnt, js(X, a, b))
            break
    if cnt == KMAX - 1:
        print 'not found, js = %.3f' % (js(X, a, b))
    x1 = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.1)
    x2 = -a[0] / a[2] - a[1] / a[2] * x1
    plt.plot(x1, x2, color='black')
    plt.show()
    return a, b


w1 = [[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], \
      [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]]
w2 = [[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], \
      [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]
w3 = [[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], \
      [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]
w4 = [[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], \
      [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]]

X = np.array(w2 + w3)
y = np.array([1] * 10 + [-1] * 10)
a, ja1 = batchRelaxtion(X, y, ['w2', 'w3'], b=0.1, eta=2, MAXCNT=10000)
a, ja5 = batchRelaxtion(X, y, ['w2', 'w3'], b=0.5, eta=2, MAXCNT=10000)
f1 = plt.plot(ja1[:100, 0], ja1[:100, 1], '-or', label='$b=0.1$')
f5 = plt.plot(ja5[:100, 0], ja5[:100, 1], '-ob', label='$b=0.5$')
plt.xlabel(r'k')
plt.ylabel(r'jr')
plt.legend(loc=1)
plt.title(r'''criterion function''')
plt.show()
