import cupy as np

buddha_kernel = np.ElementwiseKernel(
'complex64 a, complex64 c, float32 d',
'',
'''
	a = rand();
	while (a.real() < 5) {
		a += 1;
	}
''',
'calc_buddha',
False,
''
)

n = 10
a = np.zeros(n, dtype=np.complex64)
b = np.zeros(n, dtype=np.complex64)
d = np.zeros(n, dtype=np.float32)
# buddha = np.zeros((5, 5), dtype=np.int32)
buddha_kernel(a, b, d)
print(a, b)