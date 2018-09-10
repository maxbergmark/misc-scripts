import time
import numpy as np

def find_repeating(lst, count=2):
	ret = []
	counts = [None] * len(lst)
	for i in lst:
		if counts[i] is None:
			counts[i] = i
		elif i == counts[i]:
			ret += [i]
			if len(ret) == count:
				return ret

def find_repeating_fast(lst):
	n = len(lst)-2
	num_sum = -n*(n+1)//2 + np.sum(lst)
	sq_sum = -n*(n+1)*(2*n+1)//6 + np.dot(lst, lst)

	root = (sq_sum/2 - num_sum*num_sum/4)**.5
	base = num_sum / 2
	a = int(base - root)
	b = int(base + root)
	return a, b


tests = int(input())
print("Comparison to Ludisposed's solution (best case):")

for k in range(tests):
	inp = input()
	iterations = 1
	t0 = time.clock()
	for _ in range(iterations):
		test = [int(i) for i in inp.split()]
		find_repeating(test)

	test_np = np.fromstring(inp, dtype=np.int64, sep=' ')
	t1 = time.clock()

	for _ in range(iterations):
		find_repeating_fast(test_np)
	
	t2 = time.clock()
	print("Time per iteration (10^%d): %9.2fµs /%9.2fµs, speedup: %5.2f" % (
		k+1, 
		(t1-t0)/iterations*1e6,
		(t2-t1)/iterations*1e6, 
		(t1-t0)/(t2-t1))
	)

"""
for k in range(1, 7):
	size = 10**k
	iterations = 1000000//size*0+1

	# test = [i+1 for i in range(size)] + [1, size-1] # worst case
	# test = [1, 2] + [i+1 for i in range(size)] # best case
	t0 = time.clock()
	for i in range(iterations):
		find_repeating(test)
	t1 = time.clock()
	for i in range(iterations):
		test_np = np.array(test, dtype=np.int64)
		find_repeating_fast(test_np)
	t2 = time.clock()

	print("Time per iteration (10^%d): %8.2fµs /%8.2fµs, speedup: %5.2f" % (
		k, 
		(t1-t0)/iterations*1e6,
		(t2-t1)/iterations*1e6, 
		(t1-t0)/(t2-t1))
	)
"""