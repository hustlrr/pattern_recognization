# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(-1.5, -0.5, 0.1)
plt.plot(x1, 0.5 / np.abs(x1+1))  # plot返回一个列表，通过line,获取其第一个元素
x2 = np.arange(-0.5, 1, 0.1)
plt.plot(x2, 0.5 / np.abs(x2))
x3 = np.arange(1, 2.5, 0.1)
plt.plot(x3, 0.5 / np.abs(x3-2))
plt.xlabel('x')
plt.ylabel('pn(x)')
plt.show()
