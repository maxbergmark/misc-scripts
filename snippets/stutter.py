from matplotlib import pyplot as plt
import time

t0 = time.time()
n = 2000
l = [0 for i in range(n)]
for i in range(n):
	diff = time.time()-t0;
	t0 = time.time()
	l[i] = diff
	time.sleep(0.001)

plt.plot(l)
plt.show()