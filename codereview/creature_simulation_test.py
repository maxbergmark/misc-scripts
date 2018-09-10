
import math
import random
import time
def getAngle(pos1,pos2):
	dx=pos2[0]-pos1[0]
	dy=pos2[1]-pos1[1]
	rads=math.atan2(dy,dx)
	rads%=2*math.pi
	degs=math.degrees(rads)
	return degs

def getDist(pos1, pos2):
	return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
def angleDiff(source,target):
	a = target - source
	a = (a + 180) % 360 - 180
	return a
class Creature(object):
	"""A virtual creature"""
	def __init__(self):
		self.pos=[random.randint(0,500) for _ in range(2)] #random x,y
		self.heading=random.randint(0,360)
		self.vision=[0,0] #left and right relative to creature's perspective

def run_simulation(creatures):
	creatureFOV=60 #can look 60 degrees left or right
	creatureViewDist=100
	for creature in creatures:
		for otherC in creatures:
			if otherC==creature:continue #don't check own angle
			ang=angleDiff(creature.heading,getAngle(creature.pos,otherC.pos))
			if abs(ang) < creatureFOV:
				if(getDist(creature.pos,otherC.pos)<creatureViewDist):
					if ang < 0:
						creature.vision[0]=1 #left side
					else:
						creature.vision[1]=1 #right side
			if sum(creature.vision)==2:
				break #if there is already something on both sides, stop checking

creatures=[Creature() for _ in range(2000)]

t0 = time.clock()
for _ in range(1):
	run_simulation(creatures)
t1 = time.clock()
print(t1-t0)