
def get_win_percentage(n, d, p, switching):
	if d > n-p-1:
		return 0
	if not switching:
		return p/n
	return (p/n)*(n-1)/(n-d-1)

for n in range(3, 10):
	for d in range(1, n):
		for p in range(1, n):
			print(n, d, p, False, get_win_percentage(n, d, p, False))
			print(n, d, p, True,  get_win_percentage(n, d, p, True))
