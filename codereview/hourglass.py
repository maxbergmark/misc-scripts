import numpy as np
import time

input_array = np.array([[1,1,1,0,0,0],
						[0,1,0,0,0,0],
						[1,1,1,0,0,0],
						[0,0,2,4,4,0],
						[0,0,0,2,0,0],
						[0,0,1,2,4,0]])

input_array = np.array([[i+j for j in range(100)]for i in range(100)])

t0 = time.clock()
for _ in range(1000):

	# Since the I-shape has size 3x3, we want an array with two rows and 
	# two columns fewer
	shape = tuple(i-2 for i in input_array.shape)

	# This denotes the shape of the I, with [0,0] being the upper left corner
	offsets = [[0,0], [0,1], [0,2], [1,1], [2,0], [2,1], [2,2]]

	result_array = np.zeros(shape, dtype=np.int64)

	# Add all offsets to each value in result_array
	for x,y in offsets:
		result_array += input_array[x:x+shape[0], y:y+shape[1]]

	# The result_array will contain the sum of every I-shape for the input_array
	print(result_array.max())
t1 = time.clock()
print(t1-t0)