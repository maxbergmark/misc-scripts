import java.util.PrimitiveIterator;
import java.util.stream.IntStream;

public class Solution {
	/**
	 * Calculates the GCD of two numbers using the Euclidean Algorithm.
	 */
	private static int gcd(int a, int b) {
		while (b != 0) {
			int temp = b;
			b = a % b;
			a = temp;
		}
		return a;
	}

	public boolean isPower(int n) {
		PrimitiveIterator.OfInt factors =
			IntStream.concat(IntStream.of(2), IntStream.iterate(3, i -> i + 2))
					 .iterator();

		// Count the number of times each prime factor occurs
		IntStream.Builder exponents = IntStream.builder();
		int f, e;
		do {
			f = factors.nextInt();
			for (e = 0; n % f == 0; e++) {
				n /= f;
			}
			if (e > 0) {
				exponents.add(e);
			}
		} while (f < n);

		// Try to segregate the factors into equal groups with no loners.
		// If there is no GCD, then n was 1, so a=1, p=2 would work.
		int p = exponents.build().reduce(Solution::gcd).orElse(2);
		return p > 1;
	}
	
	public static int pow2(int a, int b) {
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

	public boolean isPower2(int n) {
		if (n < 3) {
			return n == 1;
		}

		for (int a = 2; a < Math.sqrt(n)+1; a++) {
			if (n % a*a == 0) {
				for (int p = (int) (Math.log(n)/Math.log(a)); p < 32; p++) {
					int result = pow2(a, p);
					if (result == n) {
						return true;
					}
					if (result > n) {
						break;
					}
				}
			}
		}
		return false;
	}

	public boolean isPower3(int n) {
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
			// System.out.println(p + "   " + pow2(low_a, p) + "   " + pow2(high_a, p));

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


	public static void main(String[] args) {
		Solution s = new Solution();
		int iterations = 10000000;
		int offset = 1;
		long t0, t1;
		/*
		0 = System.nanoTime();
		for (int i = offset; i < offset+iterations; i++) {
			// s.isPower(i);
			// System.out.println(i + " " + s.isPower(i));
		}
		t1 = System.nanoTime();
		double b0 = 1e-9*(t1-t0);
		System.out.println(b0);
		
		t0 = System.nanoTime();
		for (int i = offset; i < offset+iterations; i++) {
			s.isPower2(i);
			// System.out.println(i + " " + s.isPower(i));
		}
		t1 = System.nanoTime();
		double b1 = 1e-9*(t1-t0);
		System.out.println(b1);
		*/
		t0 = System.nanoTime();
		for (int i = offset; i < offset+iterations; i++) {
			// long t00 = System.nanoTime();
			s.isPower3(i);
			// long t01 = System.nanoTime();
			// System.out.println(i + ", " + (t01-t00));
		}
		t1 = System.nanoTime();
		double b2 = 1e-9*(t1-t0);
		System.out.println(b2);
	}
}