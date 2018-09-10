import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import random

rc('text', usetex=True)
rc('font', family='serif')

n = 70
x = np.array([8 + i/6 for i in range(n)])
y = np.array([int(5*random.random()**2)/4 for _ in x])
y = np.array([i//(n//2) for i in range(n)])
sigma = 2
kernel = np.array([1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(i - n/2)**2/(2*sigma**2)) for i in range(n)])

matr = np.zeros((n, n))
for i in range(n):
	min_i = max(0, i-n//2) 
	max_i = min(n, i+n//2)
	matr[i,min_i:max_i] = kernel[n-max_i:n-min_i]


sums = np.sum(matr, axis = 0)
y_filtered = np.dot(matr, y.transpose())/sums

plt.bar(x, y, width = 0.12)
plt.plot(x, y_filtered, 'r')
# plt.imshow(matr)
plt.title("Non-typical tee time demand for a single day")
plt.xlabel("Time (h)")
plt.ylabel("$d(t_i)$")
plt.show()