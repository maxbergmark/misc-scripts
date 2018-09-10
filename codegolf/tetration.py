import math

def f(a,b):
	if b == 1:
		return a
	return f(a, b-1)*math.log(a)

for a in range(1, 101):
	for b in range(1, 101):
		print(a, b, f(a, b))