import math
import os
import random
import re
import sys
import time



arr = [[1,1,1,0,0,0],
		[0,1,0,0,0,0],
		[1,1,1,0,0,0],
		[0,0,2,4,4,0],
		[0,0,0,2,0,0],
		[0,0,1,2,4,0]]

arr = [[i+j for j in range(100)]for i in range(100)]


t0 = time.clock()
for _ in range(1000):
	total = 0
	max_total = -1073741824


	for i in range(len(arr)):
	    for j in range(len(arr[i])):
	        if (j+2 < 100) and (i+2 < 100):
	            total = arr[i][j] + arr[i][j+1] + arr[i][j+2]+arr[i+1][j+1]+arr[i+2][j]+arr[i+2][j+1]+arr[i+2][j+2]
	            if max_total < total:

	                max_total = total
	print(max_total)
t1 = time.clock()
print(t1-t0)