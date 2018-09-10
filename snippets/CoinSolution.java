import java.util.Random;

class CoinSolution {

	static int[] changes = new int[1<<20];
	static int counted = 1;

	public static int coinChange(int[] coins, int amount) {
		if (amount == 0) return 0;
		if (amount <  0) return -1;

		int min = -1;
		for (int coin : coins) {
			int currentMin = coinChange(coins, amount - coin);

			// if amount is less than coin value
			if (currentMin >= 0) {
				min = min < 0 ? currentMin : Math.min(currentMin, min);
			}
		}

		return min < 0 ? -1 : min + 1;
	}

	public static int dynamicChange(int[] coins, int amount) {

		for (int i = counted; i <= amount; i++) {
			int minChange = Integer.MAX_VALUE/2;
			// loop over all coins which yield a non-negative remainder
			for (int j = 0; j < coins.length; j++) {
				if (coins[j] <= i) {
					if (changes[i-coins[j]] < minChange) {
						minChange = changes[i-coins[j]];
					}
				}
			}
			changes[i] = minChange + 1;
		}
		counted = amount > counted ? amount : counted;
		return changes[amount] < Integer.MAX_VALUE/2 ? changes[amount] : -1;
	}

	public static void main(String[] args) {
		int[] coins = {1, 2, 5};
		int amount = 11;
		long totalTime = 0;
		long iterations = 0;
		
		Random r = new Random();
		int low = 1;
		int high = 1000000;
		dynamicChange(coins, 1000000);

		for (amount = 1; amount < 1000000; amount++) {
			int result = r.nextInt(high-low) + low;
			long t0 = System.nanoTime();
			// int c0 = coinChange(coins, amount);
			int c0 = 0;
			long t1 = System.nanoTime();
			int c1 = dynamicChange(coins, result);
			long t2 = System.nanoTime();
			totalTime += t2-t1;
			iterations++;
			// System.out.println(amount + ": " + c0 + ", " + c1 + ", " + (c0==c1) + String.format(", %.3f times faster", (t1-t0)/(double)(t2-t1)));
		}
		System.out.println(String.format("Time per loop %.2f µs", .001*totalTime/(double)iterations));
	}
}