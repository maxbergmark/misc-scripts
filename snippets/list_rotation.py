# from collections import deque
# import time


import sympy,random
p=random.randrange(9**9)
while not sympy.isprime(p):p+=p+1
print(p)

max_prime = 0
max_p = 0
for i in range(10, 10000):
	if (i%100) == 0:
		print(i)
	p = i
	c = 0
	while c<10000 and not sympy.isprime(p):
		p+=p+1
		c += 1
		# print(i, p)
	if c != 10000 and p > max_prime:
		max_prime = p
		max_p = i
		print(max_p, max_prime)

print(max_p, max_prime)
'''
import sympy
p = 8
while not sympy.isprime(p):p=p*p+9
print(p)
'''
'''
def is_rotated(lst1, lst2):

    if len(lst1) != len(lst2):
        return False
    if lst1 == [] and lst2 == []:
        return True

    d_lst1 = deque(lst1)
    d_lst2 = deque(lst2)

    #rotate all possible rotations to find match
    for n in range(len(d_lst1)):
        d_lst2.rotate(n) 
        if d_lst2 == d_lst1:
            return True
        d_lst2.rotate(-n)
    return False


def get_partial(pattern):
	""" Calculate partial match table: String -> [Int]"""
	ret = [0]
	
	for i in range(1, len(pattern)):
		j = ret[i - 1]
		while j > 0 and pattern[j] != pattern[i]:
			j = ret[j - 1]
		ret.append(j + 1 if pattern[j] == pattern[i] else j)
	return ret
	
def search(T, P):

	partial, ret, j = get_partial(P), [], 0
	
	for i in range(len(T)):
		while j > 0 and T[i] != P[j]:
			j = partial[j - 1]
		if T[i] == P[j]: j += 1
		if j == len(P): 
			# ret.append(i - (j - 1))
			return True
			j = 0
	return False
	# return ret

def new_is_rotated(lst1, lst2):
	if len(lst1) != len(lst2):
		return False
	lst1 *= 2
	return search(lst1, lst2)

l0 = [i for i in range(100000)]
l1 = l0[10000:] + l0[:10000]
t0 = time.clock()
for i in range(1000):
	new_is_rotated(l0, l1)
t1 = time.clock()
for i in range(1000):
	is_rotated(l0, l1)
t2 = time.clock()

print(t1-t0)
print(t2-t1)
'''


