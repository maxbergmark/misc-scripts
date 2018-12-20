

def make_sieve(n):
	if n < 4:
		return [False, False, True, True][:n]
	sieve = [False, False, True, True] + [False, True]*((n-4)//2)

	for i in range(3, n, 2):
		if not sieve[i]:
			continue
		for j in range(i*i, n, 2*i):
			sieve[j] = False

	return sieve


n = 100000000
sieve = make_sieve(n)
primes = [i for i in range(n) if sieve[i]]
# print(sieve)
# print(primes)
print(primes[-1]*primes[-2])
