import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import Device
from pycuda import gpuarray
import time
import scipy.misc



code = """
#include <curand_kernel.h>
//#include <pycuda-complex.hpp>
#include <stdio.h>

#define X_MIN -1.5f
#define X_MAX 1.5f
#define Y_MIN -3.2f
#define Y_MAX 2.0f
#define X_DIM %(XDIM)s
#define Y_DIM %(YDIM)s
#define ITERS %(ITERS)s

//typedef pycuda::complex<float> cmplx;

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

__device__ void write_pixel(int idx, float px, float py, int ix, int iy,
	float *nums, unsigned int *canvas) {
	px = nums[4*idx+1];
	py = nums[4*idx];
	to_pixel(px, py, ix, iy);
	if (0 <= ix & ix < X_DIM & 0 <= iy & iy < Y_DIM) {
		canvas[iy*X_DIM + ix] += 1;
	}
}

__device__ void write_mask(int idx, float px, float py, int ix, int iy,
	float *nums, float *mask, unsigned int *counts) {
	px = nums[4*idx+2];
	py = nums[4*idx+3];
	px -= X_MIN;
	py -= Y_MIN;
	px /= X_MAX - X_MIN;
	py /= Y_MAX - Y_MIN;
	px *= X_DIM;
	py *= Y_DIM;
	ix = __float2int_rd(px);
	iy = __float2int_rd(py);
	if (0 <= ix & ix < X_DIM & 0 <= iy & iy < Y_DIM) {
		mask[iy*X_DIM + ix] = fmaxf(mask[iy*X_DIM + ix], counts[idx]+1);
	}
	if (0 <= ix-1 & ix-1 < X_DIM & 0 <= iy & iy < Y_DIM) {
		mask[iy*X_DIM + ix-1] = fmaxf(mask[iy*X_DIM + ix-1], counts[idx]+1);
	}
	if (0 <= ix+1 & ix+1 < X_DIM & 0 <= iy & iy < Y_DIM) {
		mask[iy*X_DIM + ix+1] = fmaxf(mask[iy*X_DIM + ix+1], counts[idx]+1);
	}
	if (0 <= ix & ix < X_DIM & 0 <= iy-1 & iy-1 < Y_DIM) {
		mask[(iy-1)*X_DIM + ix] = fmaxf(mask[(iy-1)*X_DIM + ix], counts[idx]+1);
	}
	if (0 <= ix & ix < X_DIM & 0 <= iy+1 & iy+1 < Y_DIM) {
		mask[(iy+1)*X_DIM + ix] = fmaxf(mask[(iy+1)*X_DIM + ix], counts[idx]+1);
	}
	if (0 <= ix-1 & ix-1 < X_DIM & 0 <= iy-1 & iy-1 < Y_DIM) {
		mask[(iy-1)*X_DIM + ix-1] = fmaxf(mask[(iy-1)*X_DIM + ix-1], counts[idx]+1);
	}
	if (0 <= ix-1 & ix-1 < X_DIM & 0 <= iy+1 & iy+1 < Y_DIM) {
		mask[(iy+1)*X_DIM + ix-1] = fmaxf(mask[(iy+1)*X_DIM + ix-1], counts[idx]+1);
	}
	if (0 <= ix+1 & ix+1 < X_DIM & 0 <= iy-1 & iy-1 < Y_DIM) {
		mask[(iy-1)*X_DIM + ix+1] = fmaxf(mask[(iy-1)*X_DIM + ix+1], counts[idx]+1);
	}
	if (0 <= ix+1 & ix+1 < X_DIM & 0 <= iy+1 & iy+1 < Y_DIM) {
		mask[(iy+1)*X_DIM + ix+1] = fmaxf(mask[(iy+1)*X_DIM + (ix+1)], counts[idx]+1);
	}
}

__device__ void generate_random_complex(float real, float imag, int idx,
	float *nums, float *dists, unsigned int *counts) {

	real *= X_MAX-X_MIN+3;
	real += X_MIN-2;
	imag *= Y_MAX-Y_MIN+0;
	imag += Y_MIN-0;

	nums[4*idx] = real;
	nums[4*idx+1] = imag;
	nums[4*idx+2] = real;
	nums[4*idx+3] = imag;
	dists[idx] = 0;
	counts[idx] = 0;
}

extern "C" {__global__ void create_sampling_mask(unsigned int *counts, float *nums,
	float *dists, float *mask) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i, ix, iy;
	float real, imag;

	if (idx < nstates) {
		curandState_t s = *states[idx];
		for(i = 0; i < 10000; i++) {

			real = curand_uniform(&s);
			imag = curand_uniform(&s);
			generate_random_complex(real, imag, idx, nums, dists, counts);

			while (counts[idx] < ITERS & dists[idx] < 25) {
				counts[idx]++;
				real = nums[4*idx]*nums[4*idx] - nums[4*idx+1]*nums[4*idx+1] + nums[4*idx+2];
				imag = 2*nums[4*idx]*nums[4*idx+1] + nums[4*idx+3];
				nums[4*idx] = real;
				nums[4*idx+1] = imag;
				dists[idx] = nums[4*idx]*nums[4*idx] + nums[4*idx+1]*nums[4*idx+1];
			}

			if (dists[idx] > 25) {
				write_mask(idx, real, imag, ix, iy, nums, mask, counts);
			}
		}
		*states[idx] = s;
	} else {
		printf("forbidden memory access %%d/%%d\\n", idx, nstates);
	}
} }

extern "C" {__global__ void buddha_kernel(unsigned int *counts, float *nums,
	float *dists, unsigned int *canvas) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j, ix, iy;
	float real, imag;

	if (idx < nstates) {
		curandState_t s = *states[idx];
		for(i = 0; i < 10000; i++) {

			real = curand_uniform(&s);
			imag = curand_uniform(&s);
			generate_random_complex(real, imag, idx, nums, dists, counts);
			to_pixel(real, imag, ix, iy);

			while (counts[idx] < ITERS & dists[idx] < 25) {
				counts[idx]++;
				real = nums[4*idx]*nums[4*idx] - nums[4*idx+1]*nums[4*idx+1] + nums[4*idx+2];
				imag = 2*nums[4*idx]*nums[4*idx+1] + nums[4*idx+3];
				nums[4*idx] = real;
				nums[4*idx+1] = imag;
				dists[idx] = nums[4*idx]*nums[4*idx] + nums[4*idx+1]*nums[4*idx+1];
			}

			if (dists[idx] > 25) {
				nums[4*idx] = 0;
				nums[4*idx+1] = 0;
				for (j = 0; j < counts[idx]+1; j++) {
					if (isnan(real)) {
						printf("isnan: %%d %%d %%d %%.2f %%.2f\\n", counts[idx], ix, iy, nums[4*idx+2], nums[4*idx+3]);
					}
					real = nums[4*idx]*nums[4*idx] - nums[4*idx+1]*nums[4*idx+1] + nums[4*idx+2];
					imag = 2*nums[4*idx]*nums[4*idx+1] + nums[4*idx+3];
					nums[4*idx] = real;
					nums[4*idx+1] = imag;
					write_pixel(idx, real, imag, ix, iy, nums, canvas);
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
	# print("\tSampling mask generated in %.2fs" % sampling_time)
	print("\tMaximum frequency: %d" % max_freq)
	print("\tMinimum frequency: %d" % min_freq)
	print("\tTotal time: %.2fs" % (elapsed_time,))
	print("\tIterations per second: %.2e" % (total_iterations / (elapsed_time),))

def format_and_save(cpu_canvas, x_dim, y_dim, threads, iters):
	cpu_canvas /= np.max(cpu_canvas)
	# cpu_mask /= np.max(cpu_mask)
	cpu_canvas.shape = (y_dim, x_dim)
	# cpu_mask.shape = (y_dim, x_dim)

	# this just makes the color gradient more visually pleasing
	cpu_canvas = np.minimum(1.1*cpu_canvas, cpu_canvas*.2+.8)

	file_name = "animation/pycuda_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, threads)
	file_name_mask = "pycuda_mask_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, threads)
	print("\n\tSaving %s..." % file_name)

	scipy.misc.toimage(cpu_canvas, cmin=0.0, cmax=1.0).save(file_name)
	# scipy.misc.toimage(cpu_mask, cmin=0.0, cmax=1.0).save(file_name_mask)
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
		options=['--use_fast_math', '-O3', '--ptxas-options=-v']
	)
	init_func = module.get_function("init_kernel")
	sampling_func = module.get_function("create_sampling_mask")
	fill_func = module.get_function("buddha_kernel")
	seed = np.int32(np.random.randint(0, 1<<31))
	init_func(seed, block=(b_s,1,1), grid=(threads,1,1))

	# initialize all numpy arrays
	samples = gpuarray.zeros(4*threads*b_s, dtype = np.float32)
	dists = gpuarray.zeros(threads*b_s, dtype = np.float32)
	counts = gpuarray.zeros(threads*b_s, dtype = np.uint32)
	canvas = gpuarray.zeros(y_dim*x_dim, dtype = np.uint32)
	# mask = gpuarray.zeros(y_dim*x_dim, dtype = np.float32)
	# t0 = time.time()
	# sampling_func(counts, samples, dists, mask, block=(b_s,1,1), grid=(threads,1,1))
	# context.synchronize()
	# t1 = time.time()
	# start calculation
	t2 = time.time()
	fill_func(counts, samples, dists, canvas, block=(b_s,1,1), grid=(threads,1,1))
	context.synchronize()
	t3 = time.time()

	# fetch buffer from gpu and save as image
	cpu_canvas = canvas.get().astype(np.float64)
	# cpu_mask = mask.get().astype(np.float64)
	context.pop()
	print_stats(cpu_canvas, t3-t2, x_dim, y_dim)
	format_and_save(cpu_canvas, x_dim, y_dim, threads, iters)

if __name__ == "__main__":

	for iters in range(43, 100):
		x_dim = 1440
		y_dim = 2560
		# iters = 20
		generate_image(x_dim, y_dim, iters)