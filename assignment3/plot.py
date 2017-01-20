# coding=utf-8

from matplotlib import pyplot as plt
import numpy as np

plt.xlim(-2, 2)
plt.ylim(-2, 2)
x = np.arange(-2, 0.6, 0.1)
plt.plot(x, 0.5 * x)
x = np.arange(0.5, 2, 0.1)
plt.plot(x, 0.5 - 0.5 * x)
x = np.arange(0.25, 2, 0.1)
plt.plot([0.5 for _ in range(x.size)], x)

xd = np.arange(-2, 0.6, 0.1)
yu = np.array([2] * xd.size)
yd = 0.5 * xd
plt.fill_between(xd, yu, yd, color='r', alpha=0.5, label='w1')
plt.fill_between(xd, yd, -yu, color='y', alpha=0.5, label='w3')
xd = np.arange(0.5, 2.1, 0.1)
yu = np.array([2] * xd.size)
yd = 0.5 - 0.5 * xd
plt.fill_between(xd, yu, yd, color='g', alpha=0.5, label='w2')
plt.fill_between(xd, yd, -yu, color='y', alpha=0.5)

plt.text(-1, 1, 'w1')
plt.text(1, 1, 'w2')
plt.text(0.5, -1, 'w3')
plt.xlabel('x1')
plt.ylabel('x2')
# 显示出来
plt.axis('scaled')
plt.show()
