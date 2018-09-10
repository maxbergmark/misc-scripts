
n = 40
coins = [1, 5, 6, 9]
values = [-1 for i in range(n)]
indices = [i for i in range(n)]
values[0] = 0

for i in range(1, n):
	mincoin = -1
	minchange = 9**9
	for coin in coins:
		if i-coin >= 0 and values[i-coin] != -1 and values[i-coin]+1 < minchange:
			minchange = values[i-coin] + 1
			mincoin = coin
	values[i] = minchange if minchange < 9**9 else -1


print(', '.join(["%2d"%i for i in values]))
print(', '.join(["%2d"%i for i in indices]))



