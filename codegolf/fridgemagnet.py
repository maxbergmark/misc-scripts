import string

subs = {l:[l] for l in string.ascii_uppercase+'0123456789'}
subs['A'].append('4')
subs['4'].append('A')

subs['B'].append('8')
subs['8'].append('B')

subs['C'].append('U')
subs['U'].append('C')

subs['E'].extend(['M', 'W', '3'])
subs['M'].extend(['E', 'W', '3'])
subs['W'].extend(['M', 'E', '3'])
subs['3'].extend(['M', 'W', 'E'])

subs['G'].extend(['6', '9'])
subs['6'].extend(['G', '9'])
subs['9'].extend(['G', '6'])

subs['I'].append('1')
subs['1'].append('I')

subs['L'].append('7')
subs['7'].append('L')

subs['N'].append('Z')
subs['Z'].append('N')

subs['O'].append('0')
subs['0'].append('O')

subs['R'].append('2')
subs['2'].append('R')

subs['S'].append('5')
subs['5'].append('S')

def check_substitution(a, b):
	a = a.replace(' ', '')
	b = b.replace(' ', '')
	if len(a) != len(b):
		return False
	for letter in a:
		for s in subs[letter]:
			if s in b:
				b = b.replace(s, '', 1)
				break
	return b == ''

print()
words = open('words.txt', 'r').read().upper().split("\n")
words = [i for i in words if len(i)>15]
print(len(words))
for i, w0 in enumerate(words):
	if len(w0) > 9 and w0.isalpha():
		# print(w0)
		for j in range(i, len(words)):
			w1 = words[j]
			if len(w0) == len(w1) and sorted(w0) != sorted(w1):
				try:
					check = check_substitution(w0, w1)
					if check:
						print('["' + w0 + '","' + w1 + '"]')
				except:
					pass


# m = [["CIRCA 333", "ICE CREAM"], ["DCLV 00133", "I LOVE CODE"], ["WE ARE EMISSARIES", "33   423    3315542135"], ["WE WANT ICE CREAM", "MET CIRCA 334 MEN"], ["I HAVE ICE CREAM", "HAVE 2 ICE CREAMS"]]
# for mess in m:
	# print('[' + mess[0] + ',' + mess[1] + '] =>', check_substitution(mess[0], mess[1]))
