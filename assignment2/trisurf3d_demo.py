from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# n_angles = 36
# n_radii = 8
#
# # An array of radii
# # Does not include radius r=0, this is to eliminate duplicate points
# radii = np.linspace(0.125, 1.0, n_radii)
#
# # An array of angles
# angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
#
# # Repeat all angles for each radius
# angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
#
# # Convert polar (radii, angles) coords to cartesian (x, y) coords
# # (0, 0) is added here. There are no duplicate points in the (x, y) plane
# x = np.append(0, (radii*np.cos(angles)).flatten())
# y = np.append(0, (radii*np.sin(angles)).flatten())

X = np.arange(0.1, 3, 0.1)
Y = np.arange(0.1, 3, 0.1)
X, Y = np.meshgrid(X, Y)
# Pringle surface
z = np.exp(-0.5 * X) * (8**-1)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('x1', fontsize=16)
ax.set_ylabel('x2', fontsize=16)
ax.set_zlabel(r'''$p\left( x_1,x_2 \right)$''', fontsize=20, rotation=0)
ax.plot_wireframe(X, Y, z)

ax.scatter(1, 1, 0, c='r')
ax.scatter(3, 3, 0, c='r')
plt.show()

z = np.exp(-X) * (3**-1)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('x1', fontsize=16)
ax.set_ylabel('x2', fontsize=16)
ax.set_zlabel(r'''$p\left( x_1,x_2 \right)$''', fontsize=20, rotation=0)
ax.plot_wireframe(X, Y, z)

ax.scatter(1, 1, 0, c='r')
ax.scatter(3, 3, 0, c='r')
plt.show()