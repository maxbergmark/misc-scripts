
def switch_code(code):
	code = code.replace("'falseundefined'", "([![]]+[][[]])")
	code = code.replace("'10'", "+!+[]+[+[]]")
	code = code.replace("'false'", "(![]+[])")
	code = code.replace("'undefined'", "([][[]]+[])")
	code = code.replace("0", "+[]")
	code = code.replace("1", "+!+[]")
	code = code.replace("2", "!+[]+!+[]")
	code = code.replace("3", "!+[]+!+[]+!+[]")
	return code
def transform_js(code):
	ret = []
	for c in code:
		if c == "f":
			temp = switch_code("'false'[0]")
		elif c == "a":
			temp = switch_code("'false'[1]")
		elif c == "l":
			temp = switch_code("'false'[2]")
		elif c == "s":
			temp = switch_code("'false'[3]")
		elif c == "u":
			temp = switch_code("'undefined'[0]")
		elif c == "n":
			temp = switch_code("'undefined'[1]")
		elif c == "d":
			temp = switch_code("'undefined'[2]")
		elif c == "e":
			temp = switch_code("'undefined'[3]")
		elif c == "i":
			temp = switch_code("'falseundefined'['10']")
		else:
			temp = "___"
		ret.append(temp)
	return "+".join(ret)

inp = "fasansfull"
print("console.log(%s);" % transform_js(inp))