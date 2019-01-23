import sys, inspect
import random
import numpy as np

# Returns a list of all bot classes which inherit from the Bot class
def get_all_bots():
	return Bot.__subclasses__()

# The parent class for all bots
class Bot:

	def __init__(self):
		pass

	def reset(self, index, bot_names):
		self.index = index
		self.bot_names = bot_names

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return True

	def decide(self, announcements):
		return True

	def recap(self, decisions):
		self.decisions = decisions

class LieBot(Bot):

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		self.choice = random.random() < 0.5
		return not self.choice

	def decide(self, announcements):
		return self.choice

class TruthBot(Bot):

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		self.choice = random.random() < 0.5
		return self.choice

	def decide(self, announcements):
		return self.choice

class TrueBot(Bot):

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return True

	def decide(self, announcements):
		return True

class FalseBot(Bot):

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return False

	def decide(self, announcements):
		return False
s = """
class FalseBot%d(Bot):

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return False

	def decide(self, announcements):
		return False
"""
for i in range(10):
	exec(s % i)
class GoForEmpty(Bot):

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return None

	def decide(self, announcements):
		truth_count = announcements.count(True)
		false_count = announcements.count(False)
		if truth_count < false_count:
			return True
		elif false_count < truth_count:
			return False
		else:
			return random.random() < 0.5

from collections import defaultdict
class FindLiars(Bot):

	def __init__(self):
		self.stats = defaultdict(lambda:[0,0,0,0])

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		self.in_game = in_game
		return True

	def decide(self, announcements):
		self.announcements = announcements
		truth_count = 0
		false_count = 0
		for i in range(len(self.bot_names)):
			bot_truth = self.stats[self.bot_names[i]]
			if sum(bot_truth) > 10:
				truth_prob = bot_truth[0] / sum(bot_truth)
				if truth_prob > 0.4 and announcements[i] != None:
					# print(self.bot_names[i], "is a truther")
					truth_count += announcements[i]
					false_count += ~announcements[i]
				elif announcements[i] != None:
					# print(self.bot_names[i], "is a liar")
					truth_count += ~announcements[i]
					false_count += announcements[i]
			elif announcements[i] != None:
				truth_count += announcements[i]
				false_count += ~announcements[i]
			else:
				truth_count += bot_truth[2] > bot_truth[3]
				false_count += bot_truth[2] < bot_truth[3]
		return truth_count <= false_count

	def recap(self, decisions):
		for i, a in enumerate(self.announcements):
			if self.in_game[i]:
				if a != None and a == decisions[i]:
					self.stats[self.bot_names[i]][0] += 1
				if a != None and a != decisions[i]:
					self.stats[self.bot_names[i]][1] += 1
				self.stats[self.bot_names[i]][2] += decisions[i]
				self.stats[self.bot_names[i]][3] += (not decisions[i])

class HalfLie(Bot):
	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return random.random() < 0.5

	def decide(self, announcements):
		return random.random() < 0.5

class FirmThenRandom(Bot):
	def decide(self, announcements):
		return random.random() < 0.5

class RandomThenFirm(Bot):
	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return random.random() < 0.5

class GoForWinAndAbandon(Bot):
	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return random.random() < 0.5

	def decide(self, announcements):
		truth_count = announcements.count(True)
		false_count = announcements.count(False)
		if truth_count < false_count:
			return False
		elif false_count < truth_count:
			return True
		else:
			return random.random() < 0.5

class GoForLikelyWin(Bot):

	def __init__(self):
		self.true_wins = 0
		self.false_wins = 0

	def announce(self, previous_announcements, previous_choices, in_game, last_round):
		return None

	def decide(self, announcements):
		if self.true_wins > self.false_wins:
			return True
		elif self.true_wins < self.false_wins:
			return False
		return random.random() < 0.5

	def recap(self, decisions):
		truth_count = decisions.count(True)
		false_count = decisions.count(False)
		if truth_count < false_count:
			self.true_wins += 1
		else:
			self.false_wins += 1


