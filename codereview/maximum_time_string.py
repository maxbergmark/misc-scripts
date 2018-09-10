import itertools
from datetime import datetime

inp = [0,9,5,5,5,5,5,5,5]

form = "%H:%M:%S"
possible = False
max_time = datetime.strptime("00:00:00", form).time()
for perm in itertools.permutations(inp, 6):
	time_string = "%d%d:%d%d:%d%d" % perm
	try:
		time_object = datetime.strptime(time_string, form).time()
		possible = True
		max_time = max(time_object, max_time)
	except: pass

if possible:
	print(max_time)
else:
	print("Impossible")