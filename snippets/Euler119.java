
public class Euler119 {

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

	public static int digitSum(int n) {
		int s = 0;
		while (n > 0) {
			s += n%10;
			n /= 10;
		}
		return s;
	}

	public static boolean isPower3(int a, int n) {
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
			if (pow2(a, p) == n) {
				return true;
			}
		}
		return false;
	}


	public static void main(String[] args) {
		int iterations = 1000000;
		int offset = 10;
		int found = 0;

		long t0 = System.nanoTime();
		for (int i = offset; found < 30; i++) {
			int s = digitSum(i);
			if (i % 10000000 == 0) {
				System.out.println("i = " + i);
			}
			if (isPower3(s, i)) {
				System.out.println((++found) + ": " + i);
			}
		}
		long t1 = System.nanoTime();
		double b2 = 1e-9*(t1-t0);
		System.out.println(b2);
	}
}