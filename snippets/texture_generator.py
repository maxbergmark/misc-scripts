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
	width = 512
	arr = np.array([[[1., 1., 1., 1.] for j in range(width)] for i in range(128)])
	for i in range(128):
		for j in range(64):
			r = (i-64)**2+(64-j)**2
			arr[i,j,3] = int(r<(64**2))
			arr[i,width-1-j,3] = int(r<(64**2))
	matplotlib.pyplot.imsave('white_button_%d.png' % (width,), arr)

generate_button()