

def find_letters(messages):
	res = ""
	for message in messages:
		for letter in message:
			in_res = res.count(letter)
			in_message = message.count(letter)

			if in_res < in_message:
				res += (in_message - in_res)*letter

	return ''.join(sorted(res.replace(" ", "")))


inp1 = ["Hello world", "Hello aliens"]
print(inp1, "=>", find_letters(inp1))
inp2 = ["foo", "bar", "baz"]
print(inp2, "=>", find_letters(inp2))
inp3 = ["Eat more tacos", "Drink more tequila"]
print(inp3, "=>", find_letters(inp3))
inp4 = ["Golfing is a fun activity", "Code should be readable"]
print(inp4, "=>", find_letters(inp4))
inp5 = ["a", "b", "c", "d", "e", "f", "g"]
print(inp5, "=>", find_letters(inp5))
