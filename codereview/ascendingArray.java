import java.util.*;

public class ascendingArray {
	public static List<Integer> getLongestAscendingSubarray(int[] a) {
		List<Integer> longestArray = new ArrayList<Integer>();
		List<Integer> currentArray = new ArrayList<Integer>();
		for (int i = 1; i < a.length; i++) {
			if(currentArray.isEmpty()) {
				currentArray.add(a[i-1]);
			}
			if (a[i]-1 == a[i-1]) {
				currentArray.add(a[i]);
			} else {
				if(longestArray.size()<currentArray.size()) {
					longestArray.clear();
					longestArray.addAll(currentArray);
				}
				currentArray.clear();
			}
		}
		return longestArray;
	}

	public static int[] getLongestAscending(int[] a) {
		int maxLength = 0;
		int maxStart = 0;
		int length = 1;
		int start = 0;
		boolean fullAscension = true;
		for (int i = 1; i < a.length; i++) {
			if (a[i]-1 == a[i-1]) {
				length++;
			} else {
				fullAscension = false;
				if (length > maxLength) {
					maxLength = length;
					maxStart = start;
				}
				length = 1;
				start = i;
			}
		}
		if (fullAscension) {
			return a;
		}
		if (length > maxLength) {
			maxLength = length;
			maxStart = start;
		}
		int[] ret = new int[maxLength];
		System.arraycopy(a, maxStart, ret, 0, maxLength);
		return ret;
	}

	public static void main(String[] args) {

		int n = 10000000;
		int[] a = new int[n];
		int m = 1;
		int c = 0;
		while (c < n) {
			for (int i = 0; i < m & c < n; i++,c++) {
				a[c] = i;
			}
			m++;
		}
		// int[] a = {6,7,1,2,3,4,5};
		long t0 = System.nanoTime();
		List<Integer> res0 = getLongestAscendingSubarray(a);
		long t1 = System.nanoTime();
		int[] res1 = getLongestAscending(a);
		long t2 = System.nanoTime();
		// System.out.println(res0);
		// System.out.println(Arrays.toString(res1));
		System.out.println((t1-t0)/(double)(t2-t1));
	}
}