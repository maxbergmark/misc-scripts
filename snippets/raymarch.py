import numpy as np
from matplotlib import pyplot as plt
import math
from multiprocessing import Pool
import time
import scipy.misc

class Sphere:

	def __init__(self, position, radius):
		self.position = position
		self.radius = radius

	def distance(self, p):
		p = p.copy()
		# pc = p.copy()
		# np = normalize(p)
		# p = p[1]*np

		# p = rotate(p)

		# p[0] = (p[0] + 3) % 6 - 3
		# p[1] = (p[1] + 3) % 6 - 3
		d = np.abs(p-self.position)
		d[0] = (d[0] + 3) % 6 - 3
		d[1] = (d[1] + 3) % 6 - 3
		return np.linalg.norm(d) - self.radius

class Cube:

	def __init__(self, position, bounds):
		self.position = position
		self.bounds = bounds

	def distance(self, p):
		p = p.copy()
		# p = (p+2) % 4 - 2
		# p[0] = (p[0] + 2) % 4 - 2

		p[1] = (p[1] + 2) % 4 - 2
		p[2] = (p[2] + 2) % 4 - 2
		d = np.abs(p-self.position) - self.bounds;
		return np.linalg.norm(np.maximum(d, 0.0))
		# + min(max(d.x,max(d.y,d.z)),0.0);

class Diff:

	def __init__(self, o1, o2):
		self.o1 = o1
		self.o2 = o2

	def distance(self, p):
		d1 = self.o1.distance(p)
		d2 = self.o2.distance(p)
		return max(-d1, d2)

def distance(p):
	# print("distance")
	return min([o.distance(p) for o in objects])

def normal(p):
	return normalize(np.array([
		distance(np.array([p[0] + 1e-6, p[1], p[2]])),
	 	distance(np.array([p[0], p[1] + 1e-6, p[2]])),
	 	distance(np.array([p[0], p[1], p[2] + 1e-6]))
 	]))

def normalize(p):
	return p / max(1e-12, np.linalg.norm(p))

def rotate(p):
	p = p.copy()
	t = p[1]*5e-2
	rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
	p[0:3:2] = np.dot(rot, p[0:3:2])
	return p

xdim = 15*16*2**2
ydim = 15*9*2**2
fov = 90
M_PI = math.pi
imageAspectRatio = xdim / ydim 
screen = np.zeros((ydim, xdim), dtype = np.float64)
objects = [Sphere(np.array([0, 6, -10]), 2), Sphere(np.array([0, 6, 10]), 2)]
# objects = [Sphere(np.array([0, 0, 0]), 4), Cube(np.array([0, 0, 0]), np.array([1, 1, 1]))]
# objects = [Diff(objects[0], objects[1])]
camera_position = np.array([0, -17, 0], dtype = np.float64)
light_pos = np.array([0, 0, 0])
max_iters = 200

def get_pixel_value(x, y):
	# print(x, y)
	v0 = camera_position.copy()
	Px = (2 * ((x + 0.5) / xdim) - 1) * math.tan(fov / 2 * M_PI / 180) * imageAspectRatio
	Pz = (1 - 2 * ((y + 0.5) / ydim)) * math.tan(fov / 2 * M_PI / 180)
	d = np.array([-Px, 1, -Pz])
	d /= np.linalg.norm(d)
	for i in range(max_iters):
		min_dist = distance(v0)
		if min_dist < 1e-7:
			n = normal(v0)
			light_vec = light_pos - v0
			# light_vec[0] = (light_vec[0] + 6) % 12 - 6
			intensity = 1 / np.linalg.norm(light_vec)**2
			light_dir = normalize(light_vec)
			# return max(0.1, np.dot(n, light_dir))
			return intensity * max(0.0, np.dot(n, light_dir))
		v0 += d * min_dist
	return 0

# def calculate_row():
	# for x in range(xdim):

inputs = [(x, y) for y in range(ydim) for x in range(xdim)]
pool = Pool(8)
t0 = time.time()
# get_pixel_value(0, 0)
screen = np.array(pool.starmap(get_pixel_value, inputs))
screen.shape = (ydim, xdim)
pool.close()
pool.join()
t1 = time.time()
# print(inputs)
# for y in range(ydim):
	# print(y)

print((screen).sum())
print((screen).sum() / (xdim*ydim))
screen /= screen.max()
print(t1-t0)
# plt.imshow(screen, cmap='gray')
scipy.misc.imsave('test.png', screen)
# plt.show()