input_list = [(1,2),(1,4),(3,5),(5,7)]
output_dict = {}

for key, value in input_list:
	#Initialize empty list if key doesn't exist
	if key not in output_dict:
		output_dict[key] = []
	# Now we know that the key always exists
	output_dict[key].append(value)

print(output_dict)
# If you want single items to be integers instead of lists
for key, value in output_dict.items():
	if len(value) == 1:
		output_dict[key] = value[0]

print(output_dict)