#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <stdio.h>

int pow2(int a, int b) {
	int re = 1;
	while (b > 0) {
		if ((b & 1) == 1) {
			re *= a;
		}
		b >>= 1;
		a *= a; 
	}
	return re;
}

bool isPower3(int n) {
	if (n < 4) {
		return n == 1;
	}

	int maxExponent = 0;
	int tempN = n;
	while (tempN > 0) {
		maxExponent++;
		tempN >>= 1;
	}
	int low_a;
	int high_a;
	int temp_a;
	int result;

	for (int p = 2; p < maxExponent+1; p++) {

		low_a = 1<<(maxExponent/p-1);
		high_a = 1<<(maxExponent/p+1);

		while (high_a-low_a > 1) {

			temp_a = (low_a+high_a)/2;
			result = pow2(temp_a, p);

			if (result == n) {
				return true;
			}
			if (result < n) {
				low_a = temp_a;
			} else {
				high_a = temp_a;
			}
		}
	}
	return false;
}


int main() {
	int iterations = 10000000;
	int offset = 1;
	std::clock_t start1 = std::clock();
	#pragma omp parallel
	{
		#pragma omp for
		for (int i = offset; i < offset+iterations; i++) {
			#pragma omp task
			{
				isPower3(i);
			}
		}
	}
	std::clock_t end1 = std::clock();
	std::cout << "Time: " << (end1 - start1) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
}

/*
int main()
{
	const int size = 256*256*4;
	double sinTable[size];
	std::clock_t start1 = std::clock();
	#pragma omp parallel for
	for(int n=0; n<size; ++n) {
		int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
		printf("%d\n", num_threads);
		sinTable[n] = std::sin(2 * M_PI * n / size);
	}
	std::clock_t end1 = std::clock();

	std::cout << "Time: " << (end1 - start1) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

}
*/