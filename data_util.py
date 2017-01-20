# coding=utf-8

import numpy as np
import struct


def loadX(X_file_src, X_file_res):
    binfile = open(X_file_src, 'rb')
    buf = binfile.read()
    index = 0
    magic, num_samples, numRows, numColumns = struct.unpack_from('>IIII', buf, index)  # 文件描述信息
    index += struct.calcsize('>IIII')

    X = struct.unpack_from('>' + str(num_samples * numRows * numColumns) + 'B', buf, index)
    X = np.reshape(X, (num_samples, numRows * numColumns))

    binfile.close()
    # row, col = X.shape
    # for i in range(row):
    #     for j in range(col):
    #         X[i][j] = 1.0 if X[i][j] else 0.0
    np.save(X_file_res, X)
    print "load imgs finished"


def loadLabels(label_file_src, label_file_res):
    train_y_file = label_file_src
    binfile = open(train_y_file, 'rb')
    buf = binfile.read()
    index = 0
    magic, num_samples = struct.unpack_from('>II', buf, index)  # 文件描述信息
    print magic, num_samples
    index += struct.calcsize('>II')

    labels = struct.unpack_from('>' + str(num_samples) + 'B', buf, index)
    labels = np.reshape(labels, [num_samples, 1])
    binfile.close()

    labels = labels.flatten()
    np.save(label_file_res, labels)
    print "load labels finished"


if __name__ == '__main__':
    X_file_src = r'MNIST\train-images.idx3-ubyte'
    X_file_res = r'MNIST\train_X'
    loadX(X_file_src, X_file_res)

    X_file_src = r'MNIST\t10k-images.idx3-ubyte'
    X_file_res = r'MNIST\test_X'
    loadX(X_file_src, X_file_res)

    # label_file_src = r'MNIST\train-labels.idx1-ubyte'
    # label_file_res = r'MNIST\train_y'
    # loadLabels(label_file_src, label_file_res)
    #
    # label_file_src = r'MNIST\t10k-labels.idx1-ubyte'
    # label_file_res = r'MNIST\test_y'
    # loadLabels(label_file_src, label_file_res)