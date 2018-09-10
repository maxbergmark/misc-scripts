import time
import math

def is_power(n):
	if (n < 3):
		return n==1

	for a in range(2, int(n**.5 + 1), 2):

		for p in range(int(math.log(n)/math.log(a)), 32):
			if a**p == n:
				return True
			if a**p > n:
				break
	return False

def is_power_new(n):
	if (n < 3):
		return n==1

	for a in range(2, int(n**.5 + 1), 2):
		if n % a*a == 0:
			for p in range(int(math.log(n)/math.log(a)), 32):
				if a**p == n:
					return True
				if a**p > n:
					break
	return False

t0 = time.clock()
for i in range(2, 100000):
	is_power(i)
	# is_power_new(i)
	# print(i, is_power(i))
	# print(i, is_power_new(i))
	# if (is_power(i) != is_power_new(i)):
		# quit()
t1 = time.clock()
print(t1-t0)

# print(is_power(36))