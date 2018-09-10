
def base_2_str(i, bits):
	s = ''
	while i > 0:
		s = '01'[i&1] + s
		i >>= 1
	s += '0'*(bits-len(s))
	return s

def remove_bits(bits, n, b):
	n_str = base_2_str(n, bits)
	z_count = n_str.count('0')
	o_count = n_str.count('1')
	if z_count <= b:
		return '1'*(bits-b)
	if o_count <= b:
		return '0'*(bits-b)
	


bits = 5
for b in range(1, bits//2+1):
	res = set()
	for n in range(2**bits):
		print(n)
		s = remove_bits(bits, n, b)
		res.add(s)
	print(res)
