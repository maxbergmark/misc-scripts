import random
import string

lower = string.ascii_lowercase + 'åäö'
upper = string.ascii_uppercase + 'ÅÄÖ'
d = {}
for i in range(len(lower)):
	d[lower[i]] = lower[i]+upper[i]
d['é'] = 'eE'
d['è'] = 'eE'
d['ç'] = 'cC'

def create_seeds(seed):
	if (len(seed) == 0):
		yield ''
		return
	for rest in create_seeds(seed[1:]):
		letters = d[seed[0]] if seed[0] in d else '__'
		yield letters[0] + rest
		yield letters[1] + rest


words = open('ordlista.txt', 'r').read().split('\n')
'''
for seed in create_seeds('maxbergmark'):
	for word in words:
		random.seed(seed)
		if ''.join(random.sample(word, len(word))) == word:
			if (len(word) > 4):
				print(word, seed)
'''
count = 0
for word in words:
	count += 1
	if len(word) < 12:
		print('\r(%d/%d)'%(count, len(words)),word, end=' '*10, flush=True)
		for seed in create_seeds(word):
			random.seed(seed)
			if ''.join(random.sample(word, len(word))) == word:
				if (len(word) > 6):
					print('\r'+word, seed, ' '*10)
