import noise
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 100
X, Y = np.meshgrid(np.linspace(-5, 5, n), np.linspace(-5, 5, n))
# noise.pnoise2(X, Y)
# map(noise.pnoise2, X, Y),
# np.array(map(noise.pnoise2, X, Y))
# np.array(list(map(noise.pnoise2, X, Y)))
# np.array(list(map(noise.pnoise2, (X, Y))))
# np.vectorize(noise.pnoise2)
f = np.vectorize(noise.pnoise2)
f(X, Y)

Z = f(X, Y, octaves = 20)
# ax = plt.axis(projection='3d')
# ax.plot(X, Y, Z)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
# ax.plot(X, Y, Z)
ax.plot_surface(X, Y, Z)
plt.show()
