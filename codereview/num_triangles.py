# minL = 5
# maxLen = 25
# for maxL in range(minL, maxLen+1):
# # for maxL in range(0, 2):
# 	i = 0
# 	i2 = 0
# 	i3 = 0
# 	for a in range(minL, maxL+1):
# 	# for a in range(12, 13):
# 		tl0 = 0
# 		tr0 = 0
# 		for b in range(a, maxL+1):
# 			i2 += min(a, maxL+1-b)
# 			if a < maxL+1-b:
# 				tl0 += a
# 			else:
# 				tr0 += maxL+1-b
# 			for c in range(b, maxL+1):
# 				if a+b > c:
# 					i += 1
# 		# print("1st:", a, tl0, tr0)
# 		# tl = max(0, a*(maxL-2*a+1))
# 		# tr = max(0, a*(maxL+1)-a*(2*maxL-a+1)//2)
# 		right_terms = maxL-max(a, maxL-a+1)+1
# 		tl = a*max(0, maxL-2*a+1)
# 		tr = right_terms*(maxL+1) - right_terms*(maxL + max(a, maxL-a+1))//2
# 		# print("2nd:", a, tl, tr, a*(maxL+1)-a*(2*maxL-a+1)//2)
# 		# print()
# 		i3 += tl+tr
# 	print(maxL, i, i2, i3)


# l = 20
# for a in range(l+1):
# 	i2 = 0
# 	# a = 230
# 	tl0 = 0
# 	tr0 = 0
# 	for b in range(a, l+1):
# 		i2 += min(a, l+1-b)
# 		if a < l+1-b:
# 			tl0 += a
# 		else:
# 			# print(l+1-b)
# 			tr0 += l+1-b

# 	tl = a*max(0, l-2*a+1)
# 	tr = (l-max(a, l-a+1)+1)*(l+1) - (l-max(a, l-a+1)+1)*(l + max(a, l-a+1))//2
# 	i3 = tl+tr
# 	# print(tl0, tr0)
# 	# print(tl, tr)
# 	print(i2, i3)

minL = 1
maxL = 2
total_ways = 0
for a in range(minL, maxL+1):
	right_terms = maxL-max(a, maxL-a+1)+1
	left_sum = a*max(0, maxL-2*a+1)
	right_sum = right_terms*(maxL+1) - right_terms*(maxL + max(a, maxL-a+1))//2
	total_ways += left_sum + right_sum
print(total_ways)