import itertools
import time

alpha = "bcdfghjklmnpqrstvwxz"
digits = "2456789"
def get_letters_greater(letter, forbidden):
	for a in alpha:
		if a > letter and a not in forbidden:
			yield a

def get_letters(forbidden = set()):
	for a in alpha:
		if a not in forbidden:
			yield a

def get_numbers():
	for d in digits:
		yield d

order = [
	0,2,3,6,8,9,12,13,14,15,16,17,24,26,27,30,32,33,36,37,38,
	39,40,41,48,50,51,54,56,57,60,61,62,63,64,65,72,73,74,75,
	76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
]

def get_permutations(s):
	all_perms = list(itertools.permutations(s))
	unique_perms = [''.join(all_perms[i]) for i in order]
	return unique_perms

def get_all_combinations():
	strings = []
	for digit in get_numbers():
		for double_letter in get_letters():
			forbidden = set(double_letter)
			for letter_1 in get_letters(forbidden):
				two_forbidden = set(double_letter + letter_1)
				for letter_2 in get_letters_greater(letter_1, two_forbidden):
					s = digit + letter_1 + letter_2 + double_letter*2
					strings.append(s)

	flat_list = [item for s in strings for item in get_permutations(s)]
	return flat_list

orders = [(0, 1, 2, 3, 3), (0, 1, 3, 2, 3), (0, 1, 3, 3, 2), (0, 2, 1, 3, 3), (0, 2, 3, 1, 3), (0, 2, 3, 3, 1), (0, 3, 1, 2, 3), (0, 3, 1, 3, 2), (0, 3, 2, 1, 3), (0, 3, 2, 3, 1), (0, 3, 3, 1, 2), (0, 3, 3, 2, 1), (1, 0, 2, 3, 3), (1, 0, 3, 2, 3), (1, 0, 3, 3, 2), (1, 2, 0, 3, 3), (1, 2, 3, 0, 3), (1, 2, 3, 3, 0), (1, 3, 0, 2, 3), (1, 3, 0, 3, 2), (1, 3, 2, 0, 3), (1, 3, 2, 3, 0), (1, 3, 3, 0, 2), (1, 3, 3, 2, 0), (2, 0, 1, 3, 3), (2, 0, 3, 1, 3), (2, 0, 3, 3, 1), (2, 1, 0, 3, 3), (2, 1, 3, 0, 3), (2, 1, 3, 3, 0), (2, 3, 0, 1, 3), (2, 3, 0, 3, 1), (2, 3, 1, 0, 3), (2, 3, 1, 3, 0), (2, 3, 3, 0, 1), (2, 3, 3, 1, 0), (3, 0, 1, 2, 3), (3, 0, 1, 3, 2), (3, 0, 2, 1, 3), (3, 0, 2, 3, 1), (3, 0, 3, 1, 2), (3, 0, 3, 2, 1), (3, 1, 0, 2, 3), (3, 1, 0, 3, 2), (3, 1, 2, 0, 3), (3, 1, 2, 3, 0), (3, 1, 3, 0, 2), (3, 1, 3, 2, 0), (3, 2, 0, 1, 3), (3, 2, 0, 3, 1), (3, 2, 1, 0, 3), (3, 2, 1, 3, 0), (3, 2, 3, 0, 1), (3, 2, 3, 1, 0), (3, 3, 0, 1, 2), (3, 3, 0, 2, 1), (3, 3, 1, 0, 2), (3, 3, 1, 2, 0), (3, 3, 2, 0, 1), (3, 3, 2, 1, 0)]


t0 = time.clock()
all_combinations = get_all_combinations()
t1 = time.clock()
print(len(all_combinations))
# print(len(set(all_combinations)))
print(t1-t0)
# print(all_combinations)

def y(n):
	s=["YBNeuoat h"[i::3].strip()for i in[0,1,2]]
	s=(s+s[1:2]+s)*n
	return ' '.join(s[:7*(n//6)+[0,2,3,4,5,7][n%6]])
def f(n):
	s = []
	for i in range(n):
		if i % 3 == 0:
			s.append("Yeah")
		if i % 2 == 0:
			s.append("But")
		if i % 3 == 1:
			s.append("No")
	return s

for i in range(1000):
	# print("%d\t%d\t%d" % (i, len(f(i)), len(f2(i))))
	print(y(i))
	# print(f(i))
# YBNBYBN