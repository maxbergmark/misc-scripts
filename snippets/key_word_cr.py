import time
import random

lst = [[random.randint(0, 100000) for i in range(3)] for _ in range(45000)]
key_words = set([i for i in range(35000)])

def get_features(sentence, key_words):
	return [word for word in sentence if word in key_words]

f = []
t0 = time.clock()
for sent in lst:
	f.append(get_features(sent, key_words))
t1 = time.clock()

print(t1-t0)