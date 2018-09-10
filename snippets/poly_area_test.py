import numpy as np
import time

def PolyArea(x,y):
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_area(x,y):
	correction = x[-1] * y[0] - y[-1]* x[0]
	main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
	return 0.5*np.abs(main_area + correction)

x = np.arange(0,1,0.001)
y = np.sqrt(1-x**2)
# print(PolyArea(x,y))
# print(polygon_area(x,y))
iters = 10000
time.sleep(.5)
t0 = time.clock()
for i in range(iters):
	polygon_area(x,y)
t1 = time.clock()
for i in range(iters):
	PolyArea(x,y)
t2 = time.clock()

print((t1-t0)/iters*1e6)
print((t2-t1)/iters*1e6)
