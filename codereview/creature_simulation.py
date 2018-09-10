import math
import random
import time

def getAngle(c1, c2):
	dx=c2.x-c1.x
	dy=c2.y-c1.y
	rads=math.atan2(dy,dx)
	return rads

def getDist(c1, c2):
	return (c1.x-c2.x)**2 + (c1.y-c2.y)**2

def angleDiff(source,target):
	a = target - source
	a = (a + math.pi) % (2*math.pi) - math.pi
	return a

class Creature(object):
	"""A virtual creature"""
	def __init__(self):
		self.x = 500*random.random()
		self.y = 500*random.random()
		self.heading=random.random()*2*math.pi
		self.vision_right = False
		self.vision_left = False
		self.FOV = 60
		self.viewDistanceSq = 100**2

def check_visibility(creature, other_creature):
	if getDist(creature, other_creature) < creature.viewDistanceSq:
		ang = angleDiff(creature.heading,getAngle(creature,other_creature))
		if abs(ang) < creature.FOV:
			if ang < 0:
				creature.vision_left = True #vision_left side
				if creature.vision_right:
					return True
			else:
				creature.vision_right = True #vision_right side
				if creature.vision_left:
					return True
	return False

def check_neighbors(creature, grid, i, j):
	for di in range(-1, 2):
		if not 0 <= i+di < 5:
			continue
		for dj in range(-1, 2):
			if not 0 <= j+dj < 5:
				continue
			for other_creature in grid[i+di][j+dj]:
				if creature == other_creature:
					continue
				checked = check_visibility(creature, other_creature)
				if checked:
					return

def run_simulation(creatures, grid):
	for creature in creatures:
		grid[int(creature.x/100)][int(creature.y/100)].append(creature)

	for i, row in enumerate(grid):
		for j, cell in enumerate(row):
			for creature in cell:
				check_neighbors(creature, grid, i, j)

creatures=[Creature() for _ in range(2000)]
t0 = time.clock()
for _ in range(1):
	grid = [[[] for i in range(5)] for j in range(6)]
	run_simulation(creatures, grid)
t1 = time.clock()
print(t1-t0)