import numpy as np
import time
import cupy as cp

def original(row, col):

	row_1 = np.asarray((col)*[0])
	for i in range(1,row):
	    row_1 = np.append(row_1, np.asarray((col)*[i]))
	row_2 = np.asarray(row*[x for x in range(0,col)])
	row_3 = np.asarray(col*(row)*[1])

	return np.vstack([row_1,row_2,row_3])

def faster(row, col):
	row1 = np.repeat(np.arange(row), col)
	row2 = np.tile(np.arange(col), row)
	row3 = np.ones_like(row2)
	return np.vstack([row1, row2, row3])


def fastest(row, col):
	arr = cp.empty((3, row*col))
	# arr[0,:] = np.repeat(np.arange(row), col)
	# arr[1,:] = np.tile(np.arange(col), row)
	# arr[2,:] = np.ones(row*col)
	return arr



row = 3000
col = 4000
n = 1

t0 = time.clock()
for _ in range(n):
	test0 = faster(row, col)
t1 = time.clock()
for _ in range(n):
	test1 = faster(row, col)
t2 = time.clock()
for _ in range(n):
	test2 = fastest(row, col)
t3 = time.clock()

print(t1-t0)
print(t2-t1)
print(t3-t2)
print()
print((t1-t0)/(t2-t1))
print((t1-t0)/(t3-t2))