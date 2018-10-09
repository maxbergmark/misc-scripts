import ast

def get_pronounce(n):
	if n < 0:
		return "minus " + get_pronounce(-n)
	words = []
	digits = ["zero", "one", "two", "three", "four",
		"five", "six", "seven", "eight", "nine"]
	for digit in str(n):
		words.append(digits[int(digit)])
	return ' '.join(words)


nums = ast.literal_eval(input())
print(sorted(nums, key=get_pronounce))
