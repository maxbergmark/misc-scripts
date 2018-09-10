
words = open('ordlista.txt', 'r').read().split('\n')

for word in words:
	sword = [ord(i) for i in sorted(word)]
	# print(word, swossrd)
	if len(word) < 2:
		continue
	diff = sword[1]-sword[0]
	is_spaced = True
	dists = []
	for i in range(1, len(word)):
		temp_dist = sword[i]-sword[i-1]-diff
		if (abs(temp_dist) > 1):
			is_spaced = False
			break
		else:
			dists.append(temp_dist+diff)
	if is_spaced and len(word)>4 and sum([abs(i-diff) for i in dists]) < 2:
		print(word, dists)