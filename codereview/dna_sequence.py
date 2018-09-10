import numpy as np
import time
import random

def transform(sequence):
	running_value = 0
	x, y = np.linspace(0, len(sequence), 2 * len(sequence) + 1), [0]
	for character in sequence:
		if character == "A":
			y.extend([running_value + 0.5, running_value])
		elif character == "C":
			y.extend([running_value - 0.5, running_value])
		elif character == "T":
			y.extend([running_value - 0.5, running_value - 1])
			running_value -= 1
		elif character == "G":
			y.extend([running_value + 0.5, running_value + 1])
			running_value += 1
		else:
			y.extend([running_value] * 2)
	return list(x), y


def transform_fast(seq, d):
	l = len(seq)
	x = np.linspace(0, l, 2 * l + 1, dtype = np.float32)
	y = np.zeros(2* l + 1, dtype = np.float32)

	atcg = np.array([d[x] for x in seq], dtype = np.int8)

	a = (atcg == 1).astype(np.int8)
	t = (atcg == 2).astype(np.int8)
	c = (atcg == 4).astype(np.int8)
	g = (atcg == 8).astype(np.int8)
	ac = a - c
	tg = -t + g

	cum_sum = np.concatenate((
		np.array([0]),
		np.cumsum(tg[:-1])
	))

	y[1::2] = cum_sum + 0.5*(ac + tg)
	y[2::2] = cum_sum + tg

	return x, y


n = 3
seq = ''.join(random.choice("ACGTUWSMKRYBDHVNZ") for _ in range(10**n))
d = {
	"A": 1, "T": 2, "C": 4, "G": 8,
	"U": 0, "W": 0, "S": 0, "M": 0,
	"K": 0, "R": 0, "Y": 0, "B": 0,
	"D": 0, "H": 0, "V": 0, "N": 0, "Z": 0
}
# seq = "ARCTG"
# seq = "ATCG"
# print(seq)
t0 = time.clock()
x0, y0 = transform(seq)
t1 = time.clock()
x1, y1 = transform_fast(seq, d)
t2 = time.clock()
print("\tBenchmark with random input string of length 10^%d" % n)
print("\ttransform(seq): %.2fms" % (1000*(t1-t0),))
print("\ttransform_fast(seq): %.2fms" % (1000*(t2-t1),))
print("\tSpeedup factor: %.2f" % ((t1-t0)/(t2-t1),))
success = True

# for i, j in zip(y0, y1):
	# if i != j:
		# success = False
		# print(i, j)

print(success)

