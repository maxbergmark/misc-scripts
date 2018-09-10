
def index_equals_value_search(arr):
	left = 0
	right = len(arr) - 1

	ind = 0
	last = -1
	while left < right:
		ind = (left + right) // 2
		if arr[ind] - ind < 0:
			left = ind + 1
		elif arr[ind] == ind:
			right = ind - 1
			last = ind
		else:
			right = ind - 1
	if arr[left] == left:
		return left
	return last

cases = [[0], [0,3], [-8,0,1,3,5], [-5,0,2,3,10,29], [-5,0,3,4,10,18,27], [-6,-5,-4,-1,1,3,5,7]]
new_case = [2*i-20 for i in range(20)]
print(new_case)

for case in cases:
	print(index_equals_value_search(case))


