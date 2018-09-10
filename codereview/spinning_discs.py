import math
import numpy

def solution(k,array):

	min = float('inf')
	initialState = numpy.array(array)
	from scipy import optimize
	array = sorted(initialState)
	o = len(initialState)

	#brute force aproach make create two array with lower and upper value
	u = [k + x for x in array]
	v = [-(k - x) for x in array]
	array = v +array + u 
	array = numpy.array(initialState,dtype = numpy.float64)

	low =array[0]
	high =array[-1]
	for i in range(o//2,initialState.size - o * 2 + o//2 + 1):
		#subarray after modify
		r =array[i:i+o]
		#median of subarray
		p = r[o//2] if o % 2 else (r[o//2] + r[o//2 - 1])/2.0
		if p > high:
			return res
		if low <= p <= high:
			step =  numpy.abs(p - r).sum()
			if step < min and p >= 0:
				#update minimum step but median is not minimum number
				min = step
				#solve equantion to make sure it is the minimum number
				def equation(x):
					x = x[0]
					return numpy.array([sum(abs(x - i) for i in r) - step])
				#found minimum step but it is not minimum number
				f = optimize.fsolve(equation, numpy.array([0]))
				res = numpy.r_[f[0],p]
				test = r
	#it is True but runs out of time
	return math.ceil(res.min())

def fast_solution(k, array):
	pass


array = [2, 7, 1]
k = 10
print(solution(k, array))
print(fast_solution(k, array))