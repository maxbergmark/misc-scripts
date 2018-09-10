import java.util.PrimitiveIterator;
import java.util.stream.IntStream;
import java.util.*;
import java.io.*;

public class PerfectPowers {
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

	public void isPower3(long n) {
		if (n < 4) {
			return;// n == 1 ? 1:-1;
		}

		int maxExponent = 0;
		long tempN = n;
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
					// System.out.print(temp_a + "^" + p + " ");
					// return temp_a;
				}
				if (result < n) {
					low_a = temp_a;
				} else {
					high_a = temp_a;
				}
			}
		}
		// return -1;
	}


	public static void main(String[] args) {
		PerfectPowers power = new PerfectPowers();
		int iterations = 10000000;
		int offset = 1;
		long t0, t1;
		List<Long> numbers = new ArrayList<Long>();

		InputStream ins = null; // raw byte-stream
		Reader r = null; // cooked reader
		BufferedReader br = null; // buffered for readLine()
		try {
		    String s;
		    ins = new FileInputStream("powers.txt");
		    r = new InputStreamReader(ins, "UTF-8"); // leave charset out for default
		    br = new BufferedReader(r);
		    while ((s = br.readLine()) != null) {
		    	try {
		    		long value = Long.valueOf(s);
			    	numbers.add(value);
			    } catch (Exception e) {}
		    }
		}
		catch (Exception e)
		{	
		    System.err.println(e.getMessage()); // handle exception
		}
		finally {
		    if (br != null) { try { br.close(); } catch(Throwable t) { /* ensure close happens */ } }
		    if (r != null) { try { r.close(); } catch(Throwable t) { /* ensure close happens */ } }
		    if (ins != null) { try { ins.close(); } catch(Throwable t) { /* ensure close happens */ } }
		}

		t0 = System.nanoTime();
		for (long i : numbers) {
			// long t00 = System.nanoTime();
			// System.out.print(i + " = ");
			power.isPower3(i);
			// System.out.println();
			// long t01 = System.nanoTime();
			// System.out.println(i + ", " + (t01-t00));
		}
		t1 = System.nanoTime();
		double b2 = 1e-9*(t1-t0);
		System.out.println(b2);
	}
}