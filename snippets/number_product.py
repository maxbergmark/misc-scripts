import time
import random
from functools import reduce
from operator import mul
import numpy as np

def prod(iterable):
    return reduce(mul, iterable, 1)

def get_products_of_all_ints_except_at_indexn2(l):
    """uses n squared mults, no divides, ie brute force"""
    if len(l) == 0:
        return []

    if len(l) == 1:
        return [1]

    return [prod(l[:i]) * prod(l[i+1:]) for i, _ in enumerate(l)]

	def get_products_fast(inp):
		prod = 1
		for i in inp:
			prod *= i
		output = [prod//i for i in inp]
		return output

	def get_products_faster(inp):
		prod = np.prod(inp)
		return prod//inp

n = 10000
# inp = [1, 7, 3, 4]
inp = np.array([random.randint(1, 10) for i in range(20)])
t0 = time.clock()
for i in range(n):
	get_products_of_all_ints_except_at_indexn2(inp)
t1 = time.clock()
for i in range(n):
	get_products_fast(inp)
t2 = time.clock()
for i in range(n):
	get_products_faster(inp)
t3 = time.clock()

print(t1-t0)
print(t2-t1)
print(t3-t2)
print((t1-t0)/(t2-t1))

print(get_products_fast(np.array([1, 7, 3, 4])))
print(get_products_faster(np.array([1, 7, 3, 4])))
