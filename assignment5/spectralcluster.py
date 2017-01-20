# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(X, gamma, neighbors):
    K = np.zeros(shape=(X.shape[0], X.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i==j:
                continue
            K[i][j] = np.linalg.norm(X[i]-X[j]) ** 2
    K *= -gamma
    np.exp(K, K)    # exponentiate K in-place
    for i in range(X.shape[0]):
        idx = K[i].argsort()[:neighbors]
        for j in range(X.shape[0]):
            if j not in idx:
                K[i][j] = 0
    return K

def spectralCluster(X, delta, k, true_):
    '''
    :param X: np.array
    :param delta: float
    :param k: int
    :return:
    '''
    W = rbf_kernel(X, delta, k)

    D = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
    L = D - W
    Lsys = np.dot(np.dot(D, L), D)
    w, v = np.linalg.eig(Lsys)  # 特征值和特征向量
    w_idx = w.argsort()[:np.unique(true_).shape[0]]  # 前k个最小的特征值的序号
    U = v[:,w_idx]
    print(U.shape)
    norm = np.linalg.norm(U, axis=1)
    T = U[:]
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T[i][j] /= norm[i]

    colors = ['red', 'blue', 'yellow', 'magenta']
    from kmeans import kmeans
    pred_ = kmeans(T, np.unique(true_).shape[0])
    # print(pred_)

    for i in range(np.unique(true_).shape[0]):
        plt.scatter(X[pred_ == i, 0], X[pred_ == i, 1], c=colors[i], label='%d' % np.sum(pred_ == i))
    acc = np.sum(pred_==true_)*1.0/len(pred_)
    acc = acc if acc > 0.5 else 1 - acc
    plt.title('Number of neighbors:%d,gamma:%.2f,acc:%.2f' % (k, delta, acc))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X = np.loadtxt(r'spectral_x.txt')
    print(X.shape)
    true_ = np.array([0 for _ in range(100)] + [1 for _ in range(100)])
    for neighbor in range(180, 210, 10):
        for d in range(10, 13, 1):
            spectralCluster(X, delta=d, k=neighbor, true_=true_)

    # from sklearn import cluster
    #
    # W = constructGraphUsingSklearn(X, delta=12, k=2)
    # spectral = cluster.SpectralClustering(n_clusters=2,
    #                                       eigen_solver='arpack',
    #                                       affinity='rbf',
    #                                       gamma=12,
    #                                       random_state=10,
    #                                       n_neighbors=2)
    # pred_ = spectral.fit_predict(X)
    # colors = ['red', 'blue', 'yellow', 'magenta']
    # for i in range(np.unique(true_).shape[0]):
    #     plt.scatter(X[pred_ == i, 0], X[pred_ == i, 1], c=colors[i], label='%d' % np.sum(pred_ == i))
    # plt.legend()
    # plt.show()