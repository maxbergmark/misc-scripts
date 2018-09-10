#include <iostream>
#include <stdio.h>
#include <math.h>
#include <ctime>
#include <string>

int* find_duplicates(int arr[], int m) {

	long n = (long) (m-2);
	long num_sum = -n*(n+1)/2;
	long sq_sum = -n*(n+1)*(2*n+1)/6;

	for (int i = 0; i < n+2; i++) {
		long v = (long) arr[i];
		num_sum += v;
		sq_sum += v*v;
	}
	int root = (int) sqrt(2*sq_sum - num_sum*num_sum);	
	int a = (int) (num_sum - root)/2;
	int b = (int) (num_sum + root)/2;
	return new int[2] {a, b};

}

int benchmark_stdin() {
	int cases;
	std::cin >> cases;
	printf("\nBenchmarking %d test cases from std::cin\n\n", cases);
	for (int i = 0; i < cases; i++) {
		clock_t start = clock();
		int n = pow(10, i+1);
		int arr[n+2];
		for (int j = 0; j < n+2; j++) {
			std::cin >> arr[j];
		}
		clock_t begin = clock();
		int* result = find_duplicates(arr, n+2);
		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		double elapsed_total = double(end - start) / CLOCKS_PER_SEC;
		printf(
			"\t%7d, %7d (%6.0fµs) (%6.0fµs) %5.2f%%\n", 
			result[0], 
			result[1], 
			elapsed_secs*1e6, 
			elapsed_total*1e6,
			100*elapsed_secs/elapsed_total
		);
	}	
}

int benchmark_fast() {
	clock_t start = clock();
	int n = 1000000;
	int test_arr[n+2];
	for (int i = 0; i < n; i++) {
		test_arr[i] = i+1;
	}
	test_arr[n] = 2;
	test_arr[n+1] = n;
	clock_t begin = clock();
	int* result = find_duplicates(test_arr, n+2);
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	double elapsed_total = double(end - start) / CLOCKS_PER_SEC;
	printf("\nBenchmarking with array created in memory\n\n");
	printf(
		"\t%7d, %7d (%6.0fµs) (%6.0fµs) %5.2f%%\n", 
		result[0], 
		result[1], 
		elapsed_secs*1e6, 
		elapsed_total*1e6,
		100*elapsed_secs/elapsed_total
	);

}

int main(int argc, char const *argv[]) {
	benchmark_fast();
	benchmark_stdin();
	return 0;
}