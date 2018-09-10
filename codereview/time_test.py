import itertools
from datetime import datetime

inp = [2, 4, 0, 0, 0, 0]

max_time = datetime.strptime("00:00:00", "%H:%M:%S").time()
possible = False

for perm in itertools.permutations(inp, 6):
	time_string = "%d%d:%d%d:%d%d" % perm
	# print(perm, time_string)
	try:
		time_object = datetime.strptime(time_string, "%H:%M:%S").time()
		possible = True
		max_time = max(max_time, time_object)
	except:
		pass
		# print("error")

if possible:
	print(max_time)
else:
	print("Impossible")