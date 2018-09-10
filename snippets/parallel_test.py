from joblib import Parallel, delayed
import multiprocessing
from hashlib import md5
from random import random

# words = open('codegolf/words.txt', 'r').read().split()
words = [str(random()) for i in range(10**7)]
def process_input(n):
	return md5(n.encode('utf-8')).hexdigest()

inputs = words
num_cores = multiprocessing.cpu_count()
print("\n\tcores:", num_cores)

results = Parallel(n_jobs = num_cores)(delayed(process_input)(i) for i in inputs)
print("\thashes created")
# print(results)
results = sorted(results)
print("\tlist sorted")
input_2 = [i for i in range(len(results)-1)]
def check_matches(n):
	m = 0
	h0 = results[n]
	h1 = results[n+1]
	for l in range(32):
		if h0[l] == h1[l]:
			m = max(m, l+1)
		else:
			break
	return m, h0, h1, n, words[n], words[n+1]

results_2 = Parallel(n_jobs = num_cores)(delayed(check_matches)(i) for i in input_2)

# print(results_2)
print("\tmaximum hash collision:", max(results_2))