# coding=utf-8

import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

train_X = np.load(r'MNIST\train_X.npy')
train_y = np.load(r'MNIST\train_y.npy')
test_X = np.load(r'MNIST\test_X.npy')
test_y = np.load(r'MNIST\test_y.npy')
print train_X.shape, train_y.shape, test_X.shape, test_y.shape

# LDF
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.55)
lda.fit(train_X, train_y)
test_ypred = lda.predict(test_X)
acc = accuracy_score(test_y, test_ypred)
print 'accuracy of LDF:%.4f' % acc

# QDF
qda = QuadraticDiscriminantAnalysis(reg_param=0.7, store_covariances=True)
qda.fit(train_X, train_y)
test_ypred = qda.predict(test_X)
acc = accuracy_score(test_y, test_ypred)
print 'accuracy of QDF:%.4f' % acc
