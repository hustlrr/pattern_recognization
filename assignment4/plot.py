# coding=utf-8

import batchUpdate
import stochasticUpdate
import numpy as np

def multiNH():
    import matplotlib.pylab as plt
    nhs = np.arange(5, 21, 1)
    import data

    X = data.w
    y = data.y

    errs_batch = []
    errs_sto = []
    for nh in nhs:
        predy, iters = batchUpdate.backpropogation(X, y, nh=nh, eta=1.0)[:2]
        err = np.sum(y.argmax(axis=1) != predy)
        print('nh=%.0f,迭代次数=%d,错分样本个数=%d(batch)' % (nh, iters, err))
        errs_batch.append(err)

        predy, iters = stochasticUpdate.backpropogation(X, y, nh=nh, eta=1.0)[:2]
        err = np.sum(y.argmax(axis=1) != predy)
        print('nh=%.0f,迭代次数=%d,错分样本个数=%d(stochastic)' % (nh, iters, err))
        errs_sto.append(err)

    plt.plot(nhs, errs_batch, '-ob', label='batch')
    plt.plot(nhs, errs_sto, '-or', label='stochastic')
    plt.legend(loc=1)
    plt.xlabel('nh')
    plt.ylabel('the number of misclassified samples')
    plt.title('The effect of the hidden layer')
    plt.show()


def multiEta():
    import matplotlib.pylab as plt
    import data

    X = data.w
    y = data.y

    errs_batch = []
    plot_iters = []
    etas = np.arange(0.5, 1.6, 0.1)
    for eta in etas:
        predy, iters = batchUpdate.backpropogation(X, y, nh=13, eta=eta)
        err = np.sum(y.argmax(axis=1) != predy)
        print('eta=%.1f,迭代次数=%d,错分样本个数=%d(batch)' % (eta, iters, err))
        errs_batch.append(err)
        if err == 0:
            plot_iters.append((eta-0.05, 1.5-1.75*eta, 'iter:%d'%iters))
    plt.plot(etas, errs_batch, '-ob')
    plt.ylim(0, 9)
    plt.xlabel('eta')
    plt.ylabel('the number of misclassified samples')
    plt.title('The effect of learning rate')
    for px, py, text in plot_iters:
        plt.text(px, py, text, fontsize = 10)
    plt.show()


def plotStep():
    import matplotlib.pylab as plt
    import data

    X = data.w
    y = data.y

    batch_predy, batch_iters, batch_jws = batchUpdate.backpropogation(X, y, nh=13, eta=0.7)
    plt.plot(range(20), batch_jws[:20], '-ob', label='batch')
    sto_predy, sto_iters, sto_jws = stochasticUpdate.backpropogation(X, y, nh=12, eta=1.0)
    plt.plot(range(20), sto_jws[:20], '-or', label='stochastic')
    plt.legend(loc=1)
    plt.xlabel('the number of iteration')
    plt.ylabel('jw')
    plt.title('The trend of criterion function')
    plt.show()

if __name__ == '__main__':
    plotStep()