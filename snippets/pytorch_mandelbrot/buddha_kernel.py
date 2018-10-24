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
#include "/usr/local/cuda/samples/common/inc/helper_math.h"

#define X_MIN (-1.5f)
#define X_MAX 1.5f
#define Y_MIN (-3.2f)
#define Y_MAX 2.0f

#define X_MIN_SAMPLE (-2.1f)
#define X_MAX_SAMPLE 1.1f
#define Y_MIN_SAMPLE (-1.8f)
#define Y_MAX_SAMPLE 1.8f

#define X_DIM %(XDIM)s
#define Y_DIM %(YDIM)s
#define ITERS %(ITERS)s

const int nstates = %(NGENERATORS)s;
__device__ curandState_t* states[nstates];

extern "C" {
__global__
void init_kernel(int seed) {

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
}
}

__device__ void to_pixel(float2 &temp, int &ix, int &iy) {
//    ix = __float2int_rd((temp.x - X_MIN) / (X_MAX - X_MIN) *  X_DIM);
//    iy = __float2int_rd((temp.y - Y_MIN) / (Y_MAX - Y_MIN) *  Y_DIM);

	temp.x -= X_MIN;
	temp.y -= Y_MIN;
	temp.x /= X_MAX - X_MIN;
	temp.y /= Y_MAX - Y_MIN;
	temp.x *= X_DIM;
	temp.y *= Y_DIM;
	ix = __float2int_rd(temp.x);
	iy = __float2int_rd(temp.y);

}
/*
__device__
void write_pixel(float2 temp, int ix, int iy,
    float4 z, unsigned int *canvas) {
//    if (X_MIN <= z.x && z.x <= X_MAX && Y_MIN <= z.y && z.y <= Y_MAX  ) {
    if (X_MIN <= z.x & z.x <= X_MAX & Y_MIN <= z.y & z.y <= Y_MAX  ) {
        temp.x = z.y;
        temp.y = z.x;
        to_pixel(temp, ix, iy);
        atomicAdd(&(canvas[iy*X_DIM + ix]), 1);
    }
}
*/

__device__
void write_pixel(float2 temp, int ix, int iy,
	float4 z, unsigned int *canvas) {
	temp.x = z.y;
	temp.y = z.x;
	to_pixel(temp, ix, iy);
//	if (0 <= ix && ix < X_DIM && 0 <= iy && iy < Y_DIM) {
	if (0 <= ix & ix < X_DIM & 0 <= iy & iy < Y_DIM) {
		atomicAdd(&(canvas[iy*X_DIM + ix]), 1);
	}
}

__device__
void generate_random_complex(float2 temp,
	float4 &z, float &dist, unsigned int &count) {

	temp.x *= X_MAX_SAMPLE-X_MIN_SAMPLE;
	temp.x += X_MIN_SAMPLE;
	temp.y *= Y_MAX_SAMPLE-Y_MIN_SAMPLE;
	temp.y += Y_MIN_SAMPLE;

	z.x = temp.x;
	z.y = temp.y;
	z.z = temp.x;
	z.w = temp.y;
	dist = 0;
	count = 0;
}

__device__
bool check_bulbs(float4 z) {
	float zw2 = z.w*z.w;
	bool main_card = !(((z.z-0.25)*(z.z-0.25)
		+ (zw2))*(((z.z-0.25)*(z.z-0.25)
		+ (zw2))+(z.z-0.25)) < 0.25* zw2);
	bool period_2 = !((z.z+1.0) * (z.z+1.0) + (zw2) < 0.0625);
	bool smaller_bulb = !((z.z+1.309)*(z.z+1.309) + zw2 < 0.00345);
	bool smaller_bottom = !((z.z+0.125)*(z.z+0.125)
		+ (z.w-0.744)*(z.w-0.744) < 0.0088);
	bool smaller_top = !((z.z+0.125)*(z.z+0.125)
		+ (z.w+0.744)*(z.w+0.744) < 0.0088);
	return main_card & period_2 & smaller_bulb & smaller_bottom & smaller_top;
}

extern "C" {
__global__
void buddha_kernel(unsigned int *canvas, int seed) {
	int idx = blockIdx.x 
		+ threadIdx.x * gridDim.x 
		+ threadIdx.y * gridDim.x * blockDim.x;
	float gridSize = 1/1024.0f;
	int i, j, ix, iy;
	float2 temp, coord;
	unsigned int count;
	float4 z;
	float dist;
	curandState_t s;
	curand_init(seed, idx, 0, &s);

	for (coord.x = 0; coord.x < 1; coord.x += gridSize) {
		for (coord.y = 0; coord.y < 1; coord.y += gridSize) {

			for(i = 0; i < 1; i++) {

				temp.x = curand_uniform(&s);
				temp.y = curand_uniform(&s);
				temp *= gridSize;
				temp += coord;

				generate_random_complex(temp, z, dist, count);
				if (check_bulbs(z)) {
					while (count < ITERS & dist < 25) {
						count++;
						temp.x = z.x*z.x - z.y*z.y + z.z;
						temp.y = 2*z.x*z.y + z.w;
						z.x = temp.x;
						z.y = temp.y;
						dist = z.x*z.x + z.y*z.y;
					}

					if (dist > 25) {
						z.x = z.z;
						z.y = z.w;
						for (j = 0; j < count; j++) {
							temp.x = z.x*z.x - z.y*z.y + z.z;
							temp.y = 2*z.x*z.y + z.w;
							z.x = temp.x;
							z.y = temp.y;
							write_pixel(temp, ix, iy, z, canvas);
						}
					}
				}
			}
			__syncthreads();
		}
	}
}
}
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
	cpu_canvas /= max(1, np.max(cpu_canvas))
	# cpu_canvas.shape = (y_dim, x_dim)
	# this just makes the color gradient more visually pleasing
	cpu_canvas = np.minimum(2.5*cpu_canvas, cpu_canvas*.2+.8)

	file_name = "pycuda_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, threads)
	print("\n\tSaving %s..." % file_name)
	scipy.misc.toimage(cpu_canvas, cmin=0.0, cmax=1.0).save(file_name)
	print("\tImage saved!\n")

def generate_image(x_dim, y_dim, iters):

	threads = 2**7
	b_s = 2**9

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
	# init_func = module.get_function("init_kernel")
	fill_func = module.get_function("buddha_kernel")
	seed = np.int32(np.random.randint(0, 1<<31))
	# init_func(seed, block=(b_s,1,1), grid=(threads,1,1))

	# initialize all numpy arrays
	# samples = gpuarray.zeros(threads*b_s, dtype = gpuarray.vec.float4)
	# dists = gpuarray.zeros(threads*b_s, dtype = np.float32)
	# counts = gpuarray.zeros(threads*b_s, dtype = np.uint32)
	canvas = gpuarray.zeros((y_dim, x_dim), dtype = np.uint32)
	t0 = time.time()
	fill_func(canvas, seed, block=(b_s,1,1), grid=(threads,1,1))
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