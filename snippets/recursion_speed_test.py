import time

def recurse (level = 0):
	if (level == 16):
		return 1
	i = recurse(level+1)
	i += recurse(level+1)
	i += recurse(level+1)
	return i

t0 = time.clock()
w = recurse()
t1 = time.clock()

v = 0
for a in range(3):
 for b in range(3):
  for c in range(3):
   for d in range(3):
    for e in range(3):
     for f in range(3):
      for g in range(3):
       for h in range(3):
        for i in range(3):
         for j in range(3):
          for k in range(3):
           for l in range(3):
            for m in range(3):
             for n in range(3):
              for o in range(3):
               for p in range(3):
                v += 1
t2 = time.clock()
x = 0
for i in range(3**16):
	x += 1
t3 = time.clock()

print(t1-t0)
print(t2-t1)
print(t3-t2)
print(w == v == x)



