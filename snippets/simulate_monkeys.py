import random
import string

class Node:

	def __init__(self, value):
		self.value = value
		self.next = None
		self.prev = None

class LinkedList:

	def __init__(self, length):
		self.start = None
		self.end = None
		self.max_length = length
		self.length = 0

	def add(self, value):
		node = Node(value)
		if self.start == None:
			self.start = node
			self.end = node
			self.length = 1
		else:
			self.end.next = node
			node.prev = self.end
			self.end = node
			self.length += 1
		if self.length > self.max_length:
			temp = self.start
			self.start = self.start.next
			self.start.prev = None
			del temp
			self.length -= 1

	def check(self, value):
		if value[0] == self.start.value and value[-1] == self.end.value:
			temp = self.start
			for i in range(self.max_length):
				if value[i] != temp.value:
					return False
				temp = temp.next
			return True
		else:
			return False

a = LinkedList(4)
a.add('a')
a.add('b')
a.add('c')
a.add('d')
check = 'aaaa'
count = 0
limit = 10**8

for _ in range(limit):
	a.add(random.choice(string.ascii_lowercase))
	if a.check(check):
		count += 1
		# print("found")

print(count, limit)