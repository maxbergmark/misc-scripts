import java.util.*;


public class CountNWaysforSteps {

	// running time O(3 ^ n) inefficient solution 
	public static int numberOfWays(int n){

		if(n < 0 ) return 0;
		if(n == 0 ) return 1;

		return numberOfWays(n-1) + numberOfWays(n-2) + numberOfWays(n-3);
	}
	//memoized solution running time O(N)
	public static int numberOfWaysMemoized(int n, Map<Integer,Integer> cache){

		if(n < 0 ) return 0;
		if(n == 0 ) return 1;

		if(cache.containsKey(n)){
			return cache.get(n);
		}

		int calculatedNumberOFWays = numberOfWaysMemoized(n-1, cache) + numberOfWaysMemoized(n-2, cache) + numberOfWaysMemoized(n-3, cache);
		cache.put(n,calculatedNumberOFWays);
		return calculatedNumberOFWays;
	}

	public static int numberOfWaysFast(int n) {
		if (n < 2) {
			return n >= 0 ? 1 : 0;
		}
		int[] cache = new int[n+1];
		cache[0] = 1;
		cache[1] = 1;
		cache[2] = 2;
		for (int i = 3; i <= n; i++) {
			cache[i] = cache[i-1] + cache[i-2] + cache[i-3];
		}
		return cache[n];
	}

	public static int numberOfWaysFaster(int n) {
		if (n < 2) {
			return n >= 0 ? 1 : 0;
		}
		int cache0 = 1;
		int cache1 = 1;
		int cache2 = 2;
		for (int i = 3; i <= n; i++) {
			int tmp = cache0 + cache1 + cache2;
			cache0 = cache1;
			cache1 = cache2;
			cache2 = tmp;
		}
		return cache2;
	}
	public static int numberOfWaysFastest(int n) {
		int cache0 = 1;
		int cache1 = 1;
		int cache2 = 2;

		while (n >= 3) {
			cache0 = cache0 + cache1 + cache2;
			cache1 = cache1 + cache2 + cache0;
			cache2 = cache2 + cache0 + cache1;
			n -= 3;
		}

		switch (n) {
			case 0: return cache0;
			case 1: return cache1;
			case 2: return cache2;
		}

		return 0;
	}

	public static void main(String...args){
		int n = 1000000;
		long t0 = System.nanoTime();
		int res0 = numberOfWaysFaster(n);
		long t1 = System.nanoTime();
		int res1 = numberOfWaysFastest(n);
		// int res1 = numberOfWaysMemoized(n, new HashMap<>());
		long t2 = System.nanoTime();
		int res2 = numberOfWaysFast(n);
		long t3 = System.nanoTime();
		// int res1 = res0;	
		System.out.println(res0);
		System.out.println(res0 == res1 && res1 == res2);
		System.out.println("numberOfWays: " + (1e-6*(t1-t0)) + "ms");
		System.out.println("numberOfWaysMemoized: " + (1e-6*(t2-t1)) + "ms");
		System.out.println("numberOfWaysFast: " + (1e-6*(t3-t2)) + "ms");
		System.out.println((t2-t1)/(t3-t2));
	}
}