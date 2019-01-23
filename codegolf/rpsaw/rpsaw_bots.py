import random

def get_all_bots():
	return Bot.__subclasses__()

class Bot:
	def __init__(self, rounds):
		self.rounds = rounds

	def decide(self, previous_own, previous_opponent, score, opponent_score):
		return random.randint(1, 5)



class DumbBot(Bot):
	pass

class WoodBot(Bot):
	def decide(self, previous_own, previous_opponent, score, opponent_score):
		return 5