# coding=utf-8

import numpy as np


class myQDF:
    def __init__(self, X_file, y_file, shrinkage=None):
        self.X = np.load(X_file).T  # 每一列代表一个样本，每一行代表一种特征
        self.y = np.load(y_file)
        self.priors = self.getPrior()
        self.means = self.getMeans()
        self.covs = self.getCovs(shrinkage=shrinkage)

    def getPrior(self):
        classes = np.unique(self.y)
        len_ = self.y.shape[0]
        priors = []
        for group in classes:
            group_len = np.sum(group == self.y) * 1.0
            priors.append(group_len / len_)
        return priors

    def getMeans(self):
        '''
        :return: list[np.array],长度等于类别数,维数等于特征数
        '''
        classes = np.unique(self.y)
        means = []
        for group in classes:
            group_X = self.X.T[group == self.y]
            means.append(np.mean(group_X, axis=0))
        return means

    def getCovs(self, shrinkage=None):
        '''
        :return: list[np.array],每个类别的协方差矩阵
        '''
        # cov = np.cov(self.X)
        covs = []
        classes = np.unique(self.y)
        # n = self.X.shape[1]
        for group in classes:
            group_X = self.X.T[group == self.y].T
            ni = group_X.shape[1]
            tmp = np.cov(group_X)
            if shrinkage:
                # tmp = ((1 - shrinkage) * ni * tmp + shrinkage * n * cov) / ((1 - shrinkage) * ni + shrinkage * n)
                I = np.eye(tmp.shape[0], tmp.shape[1])
                tmp = (1 - shrinkage) * tmp + shrinkage * I
                covs.append(tmp)
            else:
                covs.append(tmp)
        return covs

    def getDecisionParam(self):
        '''
        :return: 二次项系数a, 一次项系数b，常数项c
        '''
        a, b, c = [], [], []
        num_of_classes = np.unique(self.y).shape[0]
        for i in range(num_of_classes):
            covi = self.covs[i]
            coviInv = np.linalg.inv(covi)
            a.append(-0.5 * coviInv)
            b.append(np.dot(coviInv, self.means[i]))
            c.append(-0.5 * np.dot(np.dot(self.means[i], coviInv), self.means[i]) +
                     np.log(self.priors[i]) -
                     0.5 * np.log(np.abs(np.linalg.det(covi))))
        return a, b, c

    def predict(self, test_X_file, test_y_file):
        test_X = np.load(test_X_file).T  # 784 by 10000
        true_y = np.load(test_y_file)
        a, b, c = self.getDecisionParam()

        predict_y = []
        for i in range(test_X.shape[1]):
            x = test_X[:, i]
            probs = []
            for a2, a1, a0 in zip(a, b, c):
                probs.append(np.dot(np.dot(x.T, a2), x) + np.dot(a1.T, x) + a0)
            predict_y.append(np.argmax(probs))
        predict_y = np.array(predict_y)

        print 'QDF: accuracy = %.4f' % (np.sum(true_y == predict_y) * 1.0 / true_y.shape[0])


if __name__ == '__main__':
    X_file = r'MNIST/train_X.npy'
    y_file = r'MNIST/train_y.npy'
    qdf = myQDF(X_file, y_file, shrinkage=0.7)
    test_X_file = r'MNIST/test_X.npy'
    test_y_file = r'MNIST/test_y.npy'
    qdf.predict(test_X_file, test_y_file)
