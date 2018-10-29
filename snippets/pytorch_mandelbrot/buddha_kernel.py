import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import Device
from pycuda import gpuarray
import time
import scipy.misc

code = open("buddha_kernel.cu", "r").read()

def print_stats(cpu_canvas, elapsed_time, x_dim, y_dim):
	total_iterations = np.sum(cpu_canvas)
	max_freq = np.max(cpu_canvas)
	min_freq = np.min(cpu_canvas)
	print("\tTotal iterations: %.5e" % total_iterations)
	print("\tIterations per pixel: %.2f" % (total_iterations / (x_dim*y_dim),))
	print("\tMaximum frequency: %d" % max_freq)
	print("\tMinimum frequency: %d" % min_freq)
	print("\tTotal time: %.2fs" % (elapsed_time,))
	print("\tIterations per second: %.2e" % (total_iterations / (elapsed_time),))

def format_and_save(cpu_canvas, x_dim, y_dim, threads, iters):
	cpu_canvas /= max(1, np.max(cpu_canvas))
	cpu_canvas.shape = (y_dim, x_dim)
	# this just makes the color gradient more visually pleasing
	cpu_canvas = np.minimum(2.5*cpu_canvas, cpu_canvas*.2+.8)

	file_name = "pycuda_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, threads)
	print("\n\tSaving %s..." % file_name)
	scipy.misc.toimage(cpu_canvas, cmin=0.0, cmax=1.0).save(file_name)
	print("\tImage saved!\n")

def transform_image(canvas, x_dim, y_dim):
	new = 0*canvas
	for y in range(y_dim):
		# print(y, y_dim)
		for x in range(x_dim):
			idx = y*x_dim + x
			dim = 16
			block = x//dim*(y_dim//dim) + y//dim
			blockRow = x % dim
			blockCol = y % dim
			transformed = block*dim*dim + blockCol*dim + blockRow

			new[idx] = canvas[transformed]
	return new

def generate_image(x_dim, y_dim, iters):

	threads = 2**7
	b_s = 2**9
	grid_size = np.float32(1/256)

	device = Device(0)
	print("\n\t" + device.name(), "\n")
	context = device.make_context()

	formatted_code = code % {
		"XDIM" : x_dim,
		"YDIM" : y_dim,
		"ITERS" : iters
	}

	# generate kernel and setup random number generation
	module = SourceModule(
		formatted_code,
		no_extern_c=True,
		options=['--use_fast_math', '-O3', '--ptxas-options=-O3']
	)
	fill_func = module.get_function("buddha_kernel")
	seed = np.int32(np.random.randint(0, 1<<31))
	canvas = gpuarray.zeros(y_dim* x_dim, dtype = np.uint32)

	t0 = time.time()
	fill_func(canvas, seed, grid_size, block=(b_s,1,1), grid=(threads,1,1))
	context.synchronize()
	t1 = time.time()

	# fetch buffer from gpu and save as image
	cpu_canvas = canvas.get().astype(np.float64)
	cpu_canvas = transform_image(cpu_canvas, x_dim, y_dim)
	context.pop()
	elapsed_time = t1-t0
	print_stats(cpu_canvas, elapsed_time, x_dim, y_dim)
	format_and_save(cpu_canvas, x_dim, y_dim, threads, iters)

if __name__ == "__main__":

	x_dim = 1440*1
	y_dim = 2560*1
	iters = 20
	generate_image(x_dim, y_dim, iters)