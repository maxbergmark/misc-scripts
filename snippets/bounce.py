import matplotlib
# Remove this if your system doesn't have support for Qt5
matplotlib.use("Qt5agg")
from matplotlib import pyplot as plt
import numpy as np
import time
import random

pos = np.array([0.,5.])

p = plt.plot(pos[0], pos[1], 'o')[0]
plt.ion()

plt.show()
plt.axis([-5, 5, 0, 10])
t0 = time.time()
while True:
	elapsed = time.time()-t0
	pos[0] = 5*np.sin(elapsed)
	pos[1] = 5*np.cos(elapsed)
	p.set_xdata(pos[0])
	p.set_ydata(pos[1])
	# plt.plot(pos[0], pos[1], 'o')
	plt.axis([-5, 5, 0, 10])
	# plt.plot(random.random(), random.random(), 'o')
	plt.pause(0.01)
