public class Solution2 {

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

	public boolean isPower(int n) {
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

	public static void main(String[] args) {
		Solution s = new Solution();
		long t0 = System.nanoTime();
		for (int i = 2; i < 1000000; i++) {
			s.isPower2(i);
			// System.out.println(i + " " + s.isPower(i));
		}
		long t1 = System.nanoTime();
		System.out.println(1e-9*(t1-t0));
	}
}
