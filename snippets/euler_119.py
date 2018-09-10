import numpy as np
import time



def slow_digit_sum(i):
	s = 0
	while (i > 0):
		s += i%10
		i //= 10
	return s

digit_sums = np.array([slow_digit_sum(i) for i in range(1000)])

def fast_digit_sum(i):
	s = 0
	while (i > 0):
		s += digit_sums[i%1000]
		i //= 1000
	return s

fast_digit_sums = np.array([fast_digit_sum(i) for i in range(1000000)])

def faster_digit_sum(i):
	s = 0
	while (i > 0):
		s += fast_digit_sums[i%1000000]
		i //= 1000000
	return s

powers = np.array([[[i, j] for j in range(2, 400)] for i in range(2, 10000)])
answers = []

print('starting calculation')
t0 = time.clock()
for a, row in enumerate(powers):
	for b, n in enumerate(row):
		e = n[0]**n[1]
		if (faster_digit_sum(e) == n[0]):
			answers.append(e)

t1 = time.clock()


answers = sorted(answers)

for i, a in enumerate(answers):
	print(i+1, a)


print(t1-t0)



