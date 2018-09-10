import sys

def triangle(height):
    print()
    max = (height * 2) - 1
    mid = 0
    while max > height:
        statement = " " * max + "/" + " " * mid + "\\"
        print(statement)
        max -= 1
        mid += 2
    statement = " " * max + "/" + "_" * mid + "\\"
    max -= 1
    print(statement)
    small = 0
    while max > 0:
        statement = " " * max + "/" + " " * small + "\\" + " " * mid + "/" + " " * small + "\\"
        print(statement)
        mid -= 2
        max -= 1
        small += 2
    statement = " " * max + "/" + "_" * small + "\\" + " " * mid + "/" + "_" * small + "\\"
    print(statement)
    pass

def create_triangle(n):
	triangle = ""
	for i in range(n):
		triangle += ' '*(2*n-i-1) + '/' + ' _'[i==n-1]*2*i + '\\' + "\n"
	for i in range(n, 2*n):
		triangle += ' '*(2*n-i-1) + '/' + ' _'[i==2*n-1]*(2*i-2*n) + '\\'
		triangle += ' '*(4*n-2*i-2) + '/' + ' _'[i==2*n-1]*(2*i-2*n) + '\\'
		triangle += '\n'
	return triangle

def create_triangle_array(n):
	arr = [[' ' for i in range(4*n)] for j in range(2*n)]
	for i in range(2*n):
		arr[i][2*n-i-1] = '/'
		arr[i][2*n+i] = '\\'
	for i in range(n, 2*n):
		arr[i][i] = '\\'
		arr[i][4*n-i-1] = '/'
	for i in range(2*n-2):
		arr[n-1][n+1+i] = '_'
		arr[2*n-1][2*n+1+i] = '_'
		arr[2*n-1][1+i] = '_'
	return '\n'.join([''.join(row) for row in arr])

def create_print_array(n):
	arr = [['*' for i in range(n)] for j in range(n)]
	return '\n'.join([''.join(row) for row in arr])


# print(create_triangle(2))
# print(create_triangle(5))
print(create_triangle_array(40))
quit()
# for i in range(10):
	# print(i)
	# triangle(i)

