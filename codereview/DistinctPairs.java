import java.util.*;

class DistinctPairs {

	public static HashSet<HashSet> getPairs(int k, List<Integer> numbers) {
		HashSet<HashSet> hs = new HashSet<HashSet>();		
		for(int i =0 ; i<numbers.size();i++){
			for(int j = i; j<numbers.size();j++){
				if(Math.abs(numbers.get(j)-numbers.get(i)) == k){
					HashSet inner = new HashSet<>();
					inner.add(numbers.get(j));
					inner.add(numbers.get(i));
					hs.add(inner);
				}
			}
		}
		return hs;
	}

	private static class Pair {
		public int a;
		public int b;
		public Pair(int a, int b) {
			this.a = a;
			this.b = b;
		}
		public String toString() {
			return "("+a+","+b+")";
		}
	}

	public static List<Pair> getPairsFast(int k, List<Integer> numbers) {
		HashSet<Integer> hLow = new HashSet<>();
		HashSet<Integer> hHigh = new HashSet<>();
		List<Pair> ret = new ArrayList<>();
		for (int i : numbers) {
			hLow.add(i);
			hHigh.add(i+k);
		}
		for (int i : hHigh) {
			if (hLow.contains(i)) {
				ret.add(new Pair(i-k, i));
				// System.out.println((i-k) + "	 " + i);
			}
		}
		return ret;
	}

	public static void main(String[] a) {
		int k = 5;
		List<Integer> numbers = new ArrayList<Integer>();
		for (int i = 0; i < 100000; i++) {
			numbers.add(i);
		}
		long t0 = System.nanoTime();
		HashSet<HashSet> r0 = getPairs(k, numbers);
		long t1 = System.nanoTime();
		List<Pair> r1 = getPairsFast(k, numbers);
		long t2 = System.nanoTime();

		System.out.println(t1-t0);
		System.out.println(t2-t1);
		System.out.println((t1-t0)/(double)(t2-t1));
	}

}