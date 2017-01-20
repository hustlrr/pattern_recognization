# coding=utf-8

import numpy as np
import matplotlib.pylab as plt


def kmeans(X, k, mu_true=None):
    '''
    :param X: np.array,样本集合，每行一个样本点
    :param k: int，聚类数目
    :param mu_true: np.array,真是的聚类中心
    :return:
    '''
    # np.random.shuffle(X)
    mu = X[np.random.choice(X.shape[0], size=k), :]
    MAXITER = 100
    distance, last_distance = None, None  # 分别表示当前迭代中的点所属类别和上一次所属类别
    threshold = 1
    for _ in range(MAXITER):
        last_distance = distance
        distance = []
        mu = sorted(mu, key=lambda x: x[1])
        for i in range(X.shape[0]):  # 样本点的数目
            d = []
            for j in range(k):
                d.append(np.sum(np.square(X[i] - mu[j])))
            distance.append(np.argmin(d))
        distance = np.array(distance)
        if np.sum(distance != last_distance) < threshold:
            print('迭代次数=%d' % (_ + 1))
            break
        for j in range(k):  # 更新聚类中心点的距离
            mu[j] = np.mean(X[distance == j], axis=0)
    # colors = ['red', 'green', 'blue', 'yellow', 'magenta']
    # for j in range(k):
    #     plt.scatter(X[distance == j][:, 0], X[distance == j][:, 1], c=colors[j], label='%d' % np.sum(distance==j))
    # mu = sorted(mu, key=lambda x:x[1])
    # plt.legend()
    # plt.title('MSE:%.4f' % np.sum(np.square(mu-mu_true)) + ',Number of Iterations:%d' % (_+1))
    # plt.show()
    return distance


if __name__ == '__main__':
    X = np.loadtxt(r'X.txt', delimiter=',')
    mu_true = np.array(sorted([[1, -1], [5.5, -4.5], [1, 4], [6, 4.5], [9, 0]], key=lambda x:x[1]))
    for random_state in range(0, 50, 10):
        print('random_state=', random_state)
        np.random.seed(random_state)
        kmeans(X, 5, mu_true)