import math
import numpy as np
import random
import time

def calc_angles(points):
	x = np.reshape(points[:,0], (len(points), 1))
	y = np.reshape(points[:,1], (len(points), 1))
	dists_x = x - x.T
	dists_y = y - y.T
	hyp = (dists_x**2 + dists_y**2)**.5
	mask = (hyp>0)
	cos = np.where(mask, dists_x / hyp, 0)
	sin = np.where(mask, dists_y / hyp, 0)
	return cos, sin

def angle(pos1,pos2):
	dx=pos2[0]-pos1[0]
	dy=pos2[1]-pos1[1]
	rads=math.atan2(dy,dx)
	rads%=2*math.pi
	degs=math.degrees(rads)
	return degs

class Point(object):
	def __init__(self):
		self.pos=[random.randint(0,500) for _ in range(2)]#random x,y
		self.angles=[]
def first_test():
	points=[Point() for _ in range(1000)]
	for point in points:
		point.angles=[] #so that each frame, they will be recalculated
		for otherC in points:
			if otherC is point:continue #don't check own angle
			ang=angle(point.pos,otherC.pos)
			point.angles.append(ang)

def second_test():
	points = np.random.rand(1000, 2)
	cos, sin = calc_angles(points)

t0 = time.clock()
first_test()
t1 = time.clock()
second_test()
t2 = time.clock()
print(t1-t0)
print(t2-t1)