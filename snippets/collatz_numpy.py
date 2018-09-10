import numpy as np
from matplotlib import pyplot as plt
import time

import os
import psutil
process = psutil.Process(os.getpid())

"""
n = 10**6
d = 10**1
counter = np.zeros(n//d, dtype = int)
x = np.arange(1, 2*n+1, 2*d, dtype=int)
values = np.arange(1, 2*n+1, 2*d, dtype=int)
completed = np.zeros(n//d, dtype=np.bool)
i = 0

t0 = time.clock()
while not all(completed):
	even = (values % 2) == 0
	values = np.where(even & ~completed, values/2, values)
	values = np.where(~even & ~completed, values*3+1, values)
	counter = np.where(~completed, counter+1, counter)
	completed = (values == 1)
	if i % 10 == 0:
		t1 = time.clock()
		print("%d\t%d\t%.2e" % (process.memory_info().rss, np.sum(~completed), 10*(n//d)/(t1-t0)))
		# time.sleep(.1)
		t0 = time.clock()
	i += 1

	# print(values)

xp = np.linspace(0, n, 1000)
yp = 260.5 + xp**.43

print(counter)
plt.plot(x, counter, '.')
plt.plot(xp, yp)
plt.show()
"""
v = 1

for i in range(10000):
	if v % 3 == 1 and (v-1)//3 > 1:
		v = (v-1)//3
	else:
		v *= 2
	print(v)