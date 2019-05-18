import numpy as np
import matplotlib.pyplot

def generate_gradient():
	arr = np.array(
		[
			[
				[
					1, 
					1, 
					1, 
					int(150<np.hypot(i, j)<200)*(.5+np.arctan2(i,j)/2/np.pi)
				] for i in range(-256, 256)
			] for j in range(-256, 256)
		])
	arr[arr[:,:,3]==0] = [0, 0, 0, 0]
	matplotlib.pyplot.imsave('angular_gradient.png', arr)

def generate_button():
	width = 2048
	height = 2048
	half_height = height//2
	arr = np.array([[[1., 1., 1., 1.] for j in range(width)] for i in range(height)])
	for i in range(height):
		for j in range(half_height):
			r = (i-half_height)**2+(half_height-j)**2
			arr[i,j,3] = int(r<(half_height**2))
			arr[i,width-1-j,3] = int(r<(half_height**2))
	matplotlib.pyplot.imsave('white_button_%dx%d.png' % (width,height), arr)

generate_button()