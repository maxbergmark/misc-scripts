from random import randint

def bot_n(n):
	s = 0
	for i in range(n):
		throw = make_throw()
		if throw == 6:
			return 0
		s += throw
	return s

def while_bot(n):
	s = 0
	while s < n:
		# print(s)
		throw = make_throw()
		if throw == 6:
			return 0
		s += throw
	return s

def make_throw():
	return randint(1,6)


n = 100000
samples = 20
scores = [0 for _ in range(samples)]
while_scores = [0 for _ in range(samples)]

for i in range(samples):
	s = 0
	s_while = 0
	for _ in range(n):
		s += bot_n(i)
		s_while += while_bot(i)
	scores[i] = s/n
	while_scores[i] = s_while/n
print(scores)
print(while_scores)