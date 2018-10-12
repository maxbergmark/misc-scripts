import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import time
import scipy.misc





code = """
#include <curand_kernel.h>
#include <pycuda-complex.hpp>
#include <stdio.h>

#define X_MIN -1.5f
#define X_MAX 1.5f
#define Y_MIN -3.2f
#define Y_MAX 2.0f
#define X_DIM %(XDIM)s
#define Y_DIM %(YDIM)s

typedef pycuda::complex<float> cmplx;

const int nstates = %(NGENERATORS)s;
__device__ curandState_t* states[nstates];

extern "C" { __global__ void initkernel(int seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < nstates) {
		curandState_t* s = new curandState_t;
		if (s != 0) {
			curand_init(seed, idx, 0, s);
		}

		states[idx] = s;
	}
} }

extern "C" {__global__ void randfillkernel(int *counts, cmplx *nums, float *dists, float *canvas, int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < nstates) {
		curandState_t s = *states[idx];
		for(int i = 0; i < 100000; i++) {

			float real = curand_uniform(&s);
			float imag = curand_uniform(&s);
			real *= X_MAX-X_MIN+3;
			real += X_MIN-2;
			imag *= Y_MAX-Y_MIN+0;
			imag += Y_MIN-0;

			nums[2*idx+1] = cmplx(real, imag);


			nums[2*idx] = cmplx(0,0);
			dists[idx] = 0;
			counts[idx] = 0;

			while (counts[idx] < 20 & dists[idx] < 25) {
				counts[idx]++;
				nums[2*idx] = nums[2*idx]*nums[2*idx] + nums[2*idx+1];
				dists[idx] = nums[2*idx].real()*nums[2*idx].real()
					+ nums[2*idx].imag()*nums[2*idx].imag();

			}
			if (dists[idx] > 25) {
				nums[2*idx] = cmplx(0,0);
				for (int i = 0; i < counts[idx]; i++) {

					nums[2*idx] = nums[2*idx]*nums[2*idx] + nums[2*idx+1];

					float px = nums[2*idx].imag();
					float py = nums[2*idx].real();
					px -= X_MIN;
					py -= Y_MIN;
					px /= X_MAX - X_MIN;
					py /= Y_MAX - Y_MIN;
					px *= X_DIM;
					py *= Y_DIM;
					int ix = (int)floorf(px);
					int iy = (int)floorf(py);
					if (0 <= ix & ix < X_DIM & 0 <= iy & iy < Y_DIM) {
						canvas[iy*X_DIM + ix] += 1;
					}

				}
			}
		}
		*states[idx] = s;
	}
} }
"""

threads = 2**10
b_s = 2**3
x_dim = 1440
y_dim = 2560

mod = SourceModule(code % { "NGENERATORS" : threads, "XDIM" : x_dim, "YDIM" : y_dim }, no_extern_c=True)
init_func = mod.get_function("initkernel")
fill_func = mod.get_function("randfillkernel")
# init_func = mod.get_function("_Z10initkerneli")
# fill_func = mod.get_function("_Z14randfillkernelPfPiPbi")
seed = np.int32(np.random.randint(0, 123412341234))
init_func(seed, block=(b_s,1,1), grid=(threads,1,1))

samples = gpuarray.zeros(2*threads*b_s, dtype = np.complex64)
dists = gpuarray.zeros(threads*b_s, dtype = np.float32)
counts = gpuarray.zeros(threads*b_s, dtype = np.int32)
canvas = gpuarray.zeros(y_dim*x_dim, dtype = np.float32)

t0 = time.time()
fill_func(counts, samples, dists, canvas, np.int32(threads), block=(b_s,1,1), grid=(threads,1,1))
# print(samples)
# print((abs(samples)**2)[::2])
# print(dists)
# print(counts)

cpu_canvas = canvas.get()
t1 = time.time()
total_iterations = np.sum(cpu_canvas)
max_freq = np.max(cpu_canvas)
cpu_canvas /= max_freq
cpu_canvas.shape = (y_dim, x_dim)
print("Total iterations: %.5e" % total_iterations)
print(max_freq)
print("Total time: %.2fs" % (t1-t0,))
print("Iterations per second: %.2e" % (total_iterations / (t1-t0),))
cpu_canvas = np.minimum(1.1*cpu_canvas, cpu_canvas*.2+.8)

scipy.misc.toimage(cpu_canvas, cmin=0.0, cmax=1.0).save(
	"pycuda_%dx%d_%d.png" % (x_dim, y_dim, threads)
)


# echo "#include <pycuda-complex.hpp>
# pycuda::complex<float> cmplx;
# void randfillkernel(cmplx *nums, int *counts, bool *mask, int N) {} " | g++ -x c++ -S - -o- | grep "^_.*:$" | sed -e 's/:$//'