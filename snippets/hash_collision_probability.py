import numpy as np
from matplotlib import pyplot as plt
import time

prob = 1
bits = 40
bits2 = 9
hashes = 2**bits
n = 10**8
k = 10**5
x = np.array([i*k for i in range(n//k)])
x2 = np.array([2**i for i in range(bits2)])
y = np.array([0.0 for _ in range(n//k)])
y2 = np.array([0.0 for _ in range(n//k)])
y3 = np.array([0.0 for _ in range(n//k)])

t0 = time.clock()
# for i in range(1, n):
	# prob *= (hashes-i)/(hashes)
	# if i%k == 0:
		# y[i//k] = 1-prob
t1 = time.clock()

for j in range(1, n//k):
	i = j*k
	pow2 = np.floor(np.log2(i))
	corr = i/pow2
	print()
	lnprob1 = .5*np.log(hashes/(hashes-i))
	lnprob2 = -i
	lnprob3 = bits*(hashes-i)*np.log(2)
	lnprob3 += (i-hashes)*np.log(hashes-i)
	# lnprob3 += (i-hashes)*(pow2*np.log(2) + (bits-pow2)*np.log(2))
	lnprob2 += lnprob3
	# print(lnprob1, lnprob2, lnprob3, lnprob4)
	y2[i//k] = 1-np.exp(lnprob1)*np.exp(lnprob2)

t2 = time.clock()

for i in range(1, n//k):
	y3[i] = np.exp(-i*(i-1)/2/hashes)


print(t1-t0)
print(t2-t1)

plt.plot(x, y, 'b')
plt.plot(x, y2, 'r')
plt.plot(x, y3, 'g')
plt.show()

