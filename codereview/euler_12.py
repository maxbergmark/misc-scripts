import numpy as np
import time

def getlistofprimes(limit):
    isprime = list(range(3, limit+1, 2))
    upper_bound = limit**0.5
    for base in range(len(isprime)):
        if not isprime[base]:
            continue
        if isprime[base] >= upper_bound:
            break
        for i in range(base + (base + 1) * isprime[base], len(isprime), isprime[base]):
            isprime[i] = None
    isprime.insert(0, 2)
    return list(filter(lambda x: x, isprime))

primes = getlistofprimes(50000)

def prime_factors(n):
	prod = 1
	remain = n
	for i in primes:
		if i*i>n:
			return prod*2
		exp = 1
		while remain % i == 0:
			exp += 1
			remain //= i
		prod *= exp
		if remain == 1:
			return prod
	return prod

def euler_12(num):
	i = 0
	for n in range(1,num):
		list= []
		for j in range(1, n**2):
			if (n+i) % j == 0:
				list.append(j)
				if len(list) == 500:
					print(n+i)
					break
		i = i + n


def get_number_of_divisors(n):
	divisors = 0
	if n%2==0:
		divisors += 1
	if n%3==0:
		divisors += 1
	for i in range(1,n//4+1):
		if n%i==0:
			divisors += 1
	return divisors+1

n=12500
t0 = time.clock()
divs = np.array(list(map(prime_factors,range(n+2)))).astype(int)
t1 = time.clock()
print(1000*(t1-t0))
val = np.floor(1.5+np.arange(0,n,.5)).astype(int)
val[val%2==0] //= 2
val.shape=n,2
triangle_divs = divs[val]
triangle_nums = np.prod(val, 1)
prods = np.prod(triangle_divs,1)
prod_arg = np.argmax(prods>=500)

print("The triangle number %d has %d divisors" % (triangle_nums[prod_arg], prods[prod_arg]))
