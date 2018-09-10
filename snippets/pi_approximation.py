import numpy as np
import time
import itertools

def approxPIsquared(error):
	prev = 8
	new = 0
	n = 3
	while (True):
		new = (prev + (8 / (n * n)))
		diff = new - prev
		if (diff <= error):
			break
		prev = new
		n = n + 2
	return new


def approximate_pi_squared(tolerance=0.0001):
    polynomial_sequence = (8 / (n * n) for n in itertools.count(1, 2))
    return sum(itertools.takewhile(tolerance.__le__, polynomial_sequence))


def approx_pi_fast(decimals):
	n = int(10**(decimals/2+.5))
	eights = 8*np.ones((1,n))
	denominators = np.arange(1, 2*n, 2)**2
	pi_squared = np.sum(eights/denominators)
	return pi_squared


def approx_pi_fast_2(decimals):
	n = int(10**(decimals/2+.5))
	# eights = 8*np.ones((1,n))
	denominators = np.arange(1, 2*n, 2)**2
	pi_squared = np.sum(8/denominators)
	return pi_squared


for i in range(10, 150):
	j = i/10
	t0 = time.clock()
	# a0 = approxPIsquared(10**-j)
	a0 = approximate_pi_squared(10**-j)
	t1 = time.clock()
	a1 = approx_pi_fast_2(j)
	t2 = time.clock()

	print()
	print(j)
	print((t1-t0)/(t2-t1))
	print(abs(a0**.5-np.pi) / abs(a1**.5-np.pi))
