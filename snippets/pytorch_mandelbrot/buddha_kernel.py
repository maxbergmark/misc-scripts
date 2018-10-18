import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import Device
from pycuda import gpuarray
import time
import scipy.misc



code = """
#include <curand_kernel.h>
#include <stdio.h>

#define X_MIN -1.5f
#define X_MAX 1.5f
#define Y_MIN -3.2f
#define Y_MAX 2.0f

#define X_MIN_SAMPLE -2.1f
#define X_MAX_SAMPLE 1.5f
#define Y_MIN_SAMPLE -2.0f
#define Y_MAX_SAMPLE 2.0f

#define X_DIM %(XDIM)s
#define Y_DIM %(YDIM)s
#define ITERS %(ITERS)s

const int nstates = %(NGENERATORS)s;
__device__ curandState_t* states[nstates];

extern "C" { __global__ void init_kernel(int seed) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < nstates) {
		curandState_t* s = new curandState_t;
		if (s != 0) {
			curand_init(seed, idx, 0, s);
		}

		states[idx] = s;
	} else {
		printf("forbidden memory access %%d/%%d\\n", idx, nstates);
	}
} }

__device__ void to_pixel(float &px, float &py, int &ix, int &iy) {
	px -= X_MIN;
	py -= Y_MIN;
	px /= X_MAX - X_MIN;
	py /= Y_MAX - Y_MIN;
	px *= X_DIM;
	py *= Y_DIM;
	ix = __float2int_rd(px);
	iy = __float2int_rd(py);
}

__device__
void write_pixel(int idx, float px, float py, int ix, int iy,
	float4 *z, unsigned int *canvas) {
	px = z[idx].y;
	py = z[idx].x;
	to_pixel(px, py, ix, iy);
	if (0 <= ix & ix < X_DIM & 0 <= iy & iy < Y_DIM) {
		canvas[iy*X_DIM + ix] += 1;
	}
}

__device__
void generate_random_complex(float real, float imag, int idx,
	float4 *z, float *dists, unsigned int *counts) {

//	real *= X_MAX-X_MIN+3;
//	real += X_MIN-2;
//	imag *= Y_MAX-Y_MIN+0;
//	imag += Y_MIN-0;
	real *= X_MAX_SAMPLE-X_MIN_SAMPLE;
	real += X_MIN_SAMPLE;
	imag *= Y_MAX_SAMPLE-Y_MIN_SAMPLE;
	imag += Y_MIN_SAMPLE;

	z[idx].x = real;
	z[idx].y = imag;
	z[idx].z = real;
	z[idx].w = imag;
	dists[idx] = 0;
	counts[idx] = 0;
}

__device__
bool check_bulbs(int idx, float4 *z) {
	float zw2 = z[idx].w*z[idx].w;
	bool main_card = !(((z[idx].z-0.25)*(z[idx].z-0.25)
		+ (zw2))*(((z[idx].z-0.25)*(z[idx].z-0.25)
		+ (zw2))+(z[idx].z-0.25)) < 0.25* zw2);
	bool period_2 = !((z[idx].z+1.0) * (z[idx].z+1.0) + (zw2) < 0.0625);
	bool smaller_bulb = !((z[idx].z+1.309)*(z[idx].z+1.309) + zw2 < 0.00345);
	bool smaller_bottom = !((z[idx].z+0.125)*(z[idx].z+0.125)
		+ (z[idx].w-0.744)*(z[idx].w-0.744) < 0.0088);
	bool smaller_top = !((z[idx].z+0.125)*(z[idx].z+0.125)
		+ (z[idx].w+0.744)*(z[idx].w+0.744) < 0.0088);
	return main_card & period_2 & smaller_bulb & smaller_bottom & smaller_top;
}

extern "C" {__global__ void buddha_kernel(unsigned int *counts, float4 *z,
	float *dists, unsigned int *canvas) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j, ix, iy;
	float real, imag;//, temp0, temp1;

	if (idx < nstates) {

		curandState_t s = *states[idx];
		for(i = 0; i < 10000; i++) {

			real = curand_uniform(&s);
			imag = curand_uniform(&s);
			generate_random_complex(real, imag, idx, z, dists, counts);
			if (check_bulbs(idx, z)) {
				while (counts[idx] < ITERS & dists[idx] < 25) {
					counts[idx]++;
					real = z[idx].x*z[idx].x - z[idx].y*z[idx].y + z[idx].z;
					imag = 2*z[idx].x*z[idx].y + z[idx].w;
					z[idx].x = real;
					z[idx].y = imag;
					dists[idx] = z[idx].x*z[idx].x + z[idx].y*z[idx].y;
				}

				if (dists[idx] > 25) {
					z[idx].x = z[idx].z;
					z[idx].y = z[idx].w;
					for (j = 0; j < counts[idx]+1; j++) {
						real = z[idx].x*z[idx].x - z[idx].y*z[idx].y + z[idx].z;
						imag = 2*z[idx].x*z[idx].y + z[idx].w;
						z[idx].x = real;
						z[idx].y = imag;
						write_pixel(idx, real, imag, ix, iy, z, canvas);
					}
				}
			}
		}
		*states[idx] = s;
	} else {
		printf("forbidden memory access %%d/%%d\\n", idx, nstates);
	}
} }
"""

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
	cpu_canvas /= np.max(cpu_canvas)
	cpu_canvas.shape = (y_dim, x_dim)
	# this just makes the color gradient more visually pleasing
	cpu_canvas = np.minimum(2.5*cpu_canvas, cpu_canvas*.2+.8)

	file_name = "pycuda_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, threads)
	print("\n\tSaving %s..." % file_name)
	scipy.misc.toimage(cpu_canvas, cmin=0.0, cmax=1.0).save(file_name)
	print("\tImage saved!\n")

def generate_image(x_dim, y_dim, iters):

	threads = 2**8
	b_s = 2**8

	device = Device(0)
	print("\n\t" + device.name(), "\n")
	context = device.make_context()

	formatted_code = code % {
		"NGENERATORS" : threads*b_s,
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
	init_func = module.get_function("init_kernel")
	fill_func = module.get_function("buddha_kernel")
	seed = np.int32(np.random.randint(0, 1<<31))
	init_func(seed, block=(b_s,1,1), grid=(threads,1,1))

	# initialize all numpy arrays
	samples = gpuarray.zeros(threads*b_s, dtype = gpuarray.vec.float4)
	dists = gpuarray.zeros(threads*b_s, dtype = np.float32)
	counts = gpuarray.zeros(threads*b_s, dtype = np.uint32)
	canvas = gpuarray.zeros(y_dim*x_dim, dtype = np.uint32)
	t0 = time.time()
	fill_func(counts, samples, dists, canvas, block=(b_s,1,1), grid=(threads,1,1))
	context.synchronize()
	t1 = time.time()

	# fetch buffer from gpu and save as image
	cpu_canvas = canvas.get().astype(np.float64)
	context.pop()
	print_stats(cpu_canvas, t1-t0, x_dim, y_dim)
	format_and_save(cpu_canvas, x_dim, y_dim, threads, iters)

if __name__ == "__main__":

	x_dim = 1440
	y_dim = 2560
	iters = 20
	generate_image(x_dim, y_dim, iters)