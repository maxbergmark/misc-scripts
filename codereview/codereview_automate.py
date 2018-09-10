
def comma_code(input_list):
	if len(input_list) == 0:
		# Raise an exception rather than just printing the error
		raise ValueError('List cannot be empty')
	# Convert everything to string (could also raise an exception if not string)
	string_list = [str(i) for i in input_list]
	# Handle the trivial case
	if len(string_list) == 1:
		return string_list[0]

	# This could be even more pythonic, but it would lose its readability
	more_than_two_items = len(string_list) > 2
	first_part = ', '.join(string_list[:-2])
	optional_separator = ', ' * more_than_two_items
	last_part = ', and '.join(string_list[-2:])
	
	formatted_string = first_part + optional_separator + last_part
	return formatted_string		

def comma_code_pythonic(input_list):
	if len(input_list) == 0:
		raise ValueError('List cannot be empty')

	string_list = [str(i) for i in input_list]

	last_part = ', and '.join(string_list[-2:])
	first_part = string_list[:-2]

	formatted_string = ', '.join(first_part + [last_part])
	return formatted_string		


# Try to place non-global variables below the function
spam = ['apples', 'bananas', 'tofu', 'cats']
for i in range(5):
	try:
		print("comma_code:", comma_code(spam[:i]))
		print("comma_code_pythonic:", comma_code_pythonic(spam[:i]))
	except ValueError as e:
		print(repr(e))
