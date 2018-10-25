#include <curand_kernel.h>
#include <stdio.h>
#include "/usr/local/cuda/samples/common/inc/helper_math.h"

#define X_MIN (-1.5f)
#define X_MAX 1.5f
#define Y_MIN (-3.2f)
#define Y_MAX 2.0f

#define X_MIN_SAMPLE (-2.1f)
#define X_MAX_SAMPLE 1.1f
#define Y_MIN_SAMPLE 0.f
#define Y_MAX_SAMPLE 1.8f

#define X_DIM %(XDIM)s
#define Y_DIM %(YDIM)s
#define ITERS %(ITERS)s

__constant__ float X_SCALE = 1/(X_MAX - X_MIN) * X_DIM;
__constant__ float Y_SCALE = 1/(Y_MAX - Y_MIN) * Y_DIM;
__constant__ static float2 xy_min = (float2){X_MIN, Y_MIN};
__constant__ static float2 xy_scale = (float2){
	1/(X_MAX - X_MIN) * X_DIM,
	1/(Y_MAX - Y_MIN) * Y_DIM
};

__device__ void to_pixel(float2 &temp, int2 &ixy) {
	temp -= xy_min;
	temp *= xy_scale;
	ixy = make_int2(temp);
}

__device__
void write_pixel(float2 temp, int2 ixy,
	float4 z, unsigned int *canvas) {
	temp.x = z.y;
	temp.y = z.x;
	to_pixel(temp, ixy);
	if (0 <= ixy.x & ixy.x < X_DIM & 0 <= ixy.y & ixy.y < Y_DIM) {
		atomicAdd(&(canvas[ixy.y*X_DIM + ixy.x]), 1);
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

__device__ __forceinline__
void write_to_image(float4 z, float2 temp, int2 ixy, 
	int count, unsigned int *canvas) {
	z.x = z.z;
	z.y = z.w;
	for (int j = 0; j < count; j++) {
		temp.x = z.x*z.x - z.y*z.y + z.z;
		temp.y = 2*z.x*z.y + z.w;
		z.x = temp.x;
		z.y = temp.y;
		write_pixel(temp, ixy, z, canvas);
	}

}

extern "C" {
__global__
void buddha_kernel(unsigned int *canvas, int seed) {
	int idx = blockIdx.x 
		+ threadIdx.x * gridDim.x 
		+ threadIdx.y * gridDim.x * blockDim.x;
	float gridSize = 1/1024.0f;
	int2 ixy;
	float2 temp, coord;
	unsigned int count;
	float4 z;
	float dist;
	curandState_t s;
	curand_init(seed, idx, 0, &s);

	for (coord.x = 0; coord.x < 1; coord.x += gridSize) {
		for (coord.y = 0; coord.y < 1; coord.y += gridSize) {

			for(int i = 0; i < 10; i++) {

				temp.x = curand_uniform(&s);
				temp.y = curand_uniform(&s);
				temp *= gridSize;
				temp += coord;

				generate_random_complex(temp, z, dist, count);
				if (check_bulbs(z)) {
					while (count < ITERS & dist < 4) {
						count++;
						temp.x = z.x*z.x - z.y*z.y + z.z;
						temp.y = 2*z.x*z.y + z.w;
						z.x = temp.x;
						z.y = temp.y;
						dist = z.x*z.x + z.y*z.y;
					}

					if (dist > 4) {
						write_to_image(z, temp, ixy, count, canvas);
						z.w *= -1;
						write_to_image(z, temp, ixy, count, canvas);						 
					}
				}
			}
			__syncthreads();
		}
	}
}
}