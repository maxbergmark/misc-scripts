
x = 0
p = 3

for digit in range(0, 10000):
	power = 10**digit
	check = 10*power
	print(digit)
	for value in range(10):
		temp_value = x + value*power
		# print(temp_value)
		if temp_value**p % check == 3:
			# print(value)
			x = temp_value
			break


print(x)
print(x**p)