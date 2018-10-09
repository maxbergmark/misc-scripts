import time
import string

s = string.ascii_lowercase*10

t0 = time.clock()
for _ in range(1000):
	int.from_bytes(bytes(s, "utf-8"), "little")
t1 = time.clock()
for _ in range(1000):
	int.from_bytes(bytes(s, "utf-8"), "big")
t2 = time.clock()
for _ in range(1000):
	sum(int(n)*(1<<(8*i)) for i, n in enumerate(bytes(s, "utf-8")[::1]))
t3 = time.clock()
for _ in range(1000):
	sum(int(n)*(1<<(8*i)) for i, n in enumerate(bytes(s, "utf-8")[::-1]))
t4 = time.clock()

print(t1-t0)
print(t2-t1)
print(t3-t2)
print(t4-t3)
