import time
import string

def first_try():
	p = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
	p_l = len(p)
	a = [98, 101, 114, 103]#, 109, 97, 114, 107]
	l = len(a)

	mods = [71, 73, 79, 83, 89, 97]
	mods = [i for i in range(70, 100)]
	total_mods = sum(mods)
	# exp_limit = 71
	add_limit = 71
	mul_limit = 71	
	found = 0
	solutions = []
	# total = exp_limit*add_limit*mul_limit
	# per_loop = add_limit*mul_limit
	# print("Expected matches: %.3f" % (71**-l * total,))

	loops = 0
	t0 = time.clock()


	for mod in mods:
		exp_limit = mod
		for exp in range(1, exp_limit+1):
			allowed = [True for _ in range(mul_limit+1)]
			for m0 in range(1, mul_limit+1):
				for m1 in range(m0+1, mul_limit+1):
					if pow(m1, exp, mod) == pow(m0, exp, mod):
						allowed[m1] = False
			for min_char in range(40, 46):
				for mul0 in range(1, mul_limit+1):
					if allowed[mul0]:
						for add0 in range(add_limit):
							vals = [pow(add0 + mul0*i, exp, mod) + min_char for i in range(2*l)]
							if vals[:l] == a:
								solutions.append(vals)
								found += 1
								print("\nfound", found, "(%d + %d*i)^%d %% %d + %d" % (add0, mul0, exp, mod, min_char))

			t1 = time.clock()
			loops += 1
			print("(%5d/%5d) Time remaining: %.3fs    "%(loops, total_mods, (total_mods/loops-1)*(t1-t0)), end='\r')

	print("\nMatches found: %d"%found)
	[print(i) for i in solutions]

def second_try():
	s = set()
	length = 5
	for a in range(800):
		# print(a)
		print(a, len(s), len(s)/26**5)
		for b in range(10):
			for c in range(20000):
				start = a
				ans = string.ascii_lowercase[start%26]
				for i in range(4):
					start = pow(start, b, 29) + c
					ans += string.ascii_lowercase[start%26]
					# print(start)
				# print(a, b, c, ans)
				s.add(ans)
				if (ans == "lunch"):
					print(a, b, c)


second_try()