
import time

def generate_partitions(k, n, prev = 0):
	if k == 1 and n > prev:
		yield (n,)
		return
	for i in range(prev+1, n//k+1):
		for rest in generate_partitions(k-1, n-i, i):
			yield (i, *rest)

n = 10000
numbers = [0 for _ in range(n)]
i = 0
while i*i < n:
	numbers[i*i] = 1
	i += 1

count = 0

t0 = time.clock()
for partition in generate_partitions(10, 200):
	# print(partition)
	num_squares = sum(numbers[i] for i in partition)
	if num_squares >= 2:
		count += 1

t1 = time.clock()
print(count)
print(t1-t0)
