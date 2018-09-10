from matplotlib import pyplot as plt
import numpy as np

def generate_gradient():
	xd, yd = 2560, 1440
	values = np.array([[-2+4*i/xd + 1.5j - 3j*j/yd for j in range(yd)]for i in range(xd)])
	original = np.copy(values)*0 + -0.4+0.6j
	iterations = np.zeros((xd, yd))
	for _ in range(100):
		values = values**2 + original
		iterations += np.abs(values) < 3
	plt.imshow(iterations.T)
	plt.show()

def generate_button():
	pass