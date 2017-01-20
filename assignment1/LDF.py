# coding=utf-8

import numpy as np


class myLDF:
    def __init__(self, X_file, y_file, shrinkage=None):
        self.X = np.load(X_file).T  # 每一列代表一个样本，每一行代表一种特征
        self.y = np.load(y_file)
        self.priors = self.getPrior()
        self.means = self.getMeans()
        self.cov = self.getCov(shrinkage=shrinkage)

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

    def getCov(self, shrinkage=None):
        '''
        :return: np.array,特征数by特征数大小的协方差矩阵
        '''
        cov = np.cov(self.X)
        if shrinkage:
            I = np.eye(cov.shape[0], cov.shape[1])
            cov = (1 - shrinkage) * cov + shrinkage * I
        return cov

    def getDecisionParam(self):
        '''
        :return: 每个类别的决策函数的系数和偏移
        '''
        coef = []
        bias = []
        num_of_classes = np.unique(self.y).shape[0]
        covInv = np.linalg.inv(self.cov)
        for i in range(num_of_classes):
            coef.append(np.dot(covInv, self.means[i]))
            bias.append(-0.5 * np.dot(np.dot(self.means[i], covInv), self.means[i]) + np.log(self.priors[i]))
        return coef, bias

    def predict(self, test_X_file, test_y_file):
        test_X = np.load(test_X_file).T  # 784 by 10000
        true_y = np.load(test_y_file)

        coef, bias = self.getDecisionParam()
        coef = np.array(coef)
        bias = np.array(bias)
        bias_ = np.zeros(shape=(bias.shape[0], true_y.shape[0]))
        for i in range(true_y.shape[0]):
            bias_[:, i] = bias.T[:]
        bias = bias_

        print coef.shape, bias.shape
        probs = np.dot(coef, test_X) + bias
        probs = probs.T
        predict_y = []
        for i in range(probs.shape[0]):
            predict_y.append(probs[i].argmax())
        predict_y = np.array(predict_y)

        print 'LDF: accuracy = %.4f' % (np.sum(true_y == predict_y) * 1.0 / true_y.shape[0])


if __name__ == '__main__':
    X_file = r'MNIST/train_X.npy'
    y_file = r'MNIST/train_y.npy'
    ldf = myLDF(X_file, y_file, shrinkage=0.01)
    test_X_file = r'MNIST/test_X.npy'
    test_y_file = r'MNIST/test_y.npy'
    ldf.predict(test_X_file, test_y_file)