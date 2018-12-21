import sys, inspect
import random

# Returns a list of all bot classes which inherit from the Bot class
def get_all_bots():
	return Bot.__subclasses__()

# The parent class for all bots
class Bot:

	def __init__(self, index, end_score):
		self.index = index
		self.end_score = end_score

	def update_state(self, current_throws):
		self.current_throws = current_throws

	def make_throw(self, scores, last_round):
		yield False

"""
# bots that throw a fixed number of times
class ThrowOnceBot(Bot):

	def make_throw(self, scores, last_round):
		yield False

class ThrowTwiceBot(Bot):

	def make_throw(self, scores, last_round):
		yield True
		yield False

class ThrowThriceBot(Bot):

	def make_throw(self, scores, last_round):
		yield True
		yield True
		yield False

class ThrowFourBot(Bot):

	def make_throw(self, scores, last_round):
		yield True
		yield True
		yield True
		yield False

class ThrowFiveBot(Bot):

	def make_throw(self, scores, last_round):
		yield True
		yield True
		yield True
		yield True
		yield False

class ThrowSixBot(Bot):

	def make_throw(self, scores, last_round):
		yield True
		yield True
		yield True
		yield True
		yield True
		yield False

class ThrowSevenBot(Bot):

	def make_throw(self, scores, last_round):
		yield True
		yield True
		yield True
		yield True
		yield True
		yield True
		yield False

# bots that aim for a fixed score

class GoToFiveBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 5:
			yield True
		yield False

class GoToSixBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 6:
			yield True
		yield False

class GoToSevenBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 7:
			yield True
		yield False

class GoToEightBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 8:
			yield True
		yield False

class GoToNineBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 9:
			yield True
		yield False

class GoToTenBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 10:
			yield True
		yield False

class GoToTwelveBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 12:
			yield True
		yield False

class GoToFifteenBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 15:
			yield True
		yield False

class GoToSixteenBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 16:
			yield True
		yield False

# advanced bots

class GoToSixteenBotSmart(Bot):

	def make_throw(self, scores, last_round):
		if not last_round:
			while sum(self.current_throws) < 16:
				yield True
			yield False
		else:
			temp_score = scores[self.index] + self.current_throws[-1]
			while temp_score < max(scores):
				yield True
				temp_score += self.current_throws[-1]


class GoTo16BotSmarter(Bot):

	def make_throw(self, scores, last_round):
		if not last_round:
			temp_score = sum(self.current_throws)
			temp_total = scores[self.index] + temp_score
			if scores[self.index] < self.end_score - 5:
				while temp_score < 16 and temp_total < self.end_score - 5:
					yield True
					temp_score = sum(self.current_throws)
					temp_total = scores[self.index] + temp_score
				yield False
			else:
				if scores[self.index] == max(scores):
					while temp_score < 10:
						yield True
						temp_score = sum(self.current_throws)
					yield False
				else:
					while temp_score < 16:
						yield True
						temp_score = sum(self.current_throws)
					yield False

		else:
			temp_score = scores[self.index] + self.current_throws[-1]
			while temp_score < max(scores):
				yield True
				temp_score += self.current_throws[-1]

"""

# testing bots

class ThrowTwiceBot(Bot):

	def make_throw(self, scores, last_round):
		yield True
		yield False

class GoToTenBot(Bot):

	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 10:
			yield True
		yield False

# PPCG bots

class AdaptiveRoller(Bot):

	def make_throw(self, scores, last_round):
		lim = min(self.end_score - scores[self.index], 22)
		while sum(self.current_throws) < lim:
			yield True
		if max(scores) == scores[self.index] and max(scores) >= self.end_score:
			yield True
		while last_round and scores[self.index] + sum(self.current_throws) <= max(scores):
			yield True
		yield False

class GoTo20Bot(Bot):

	def make_throw(self, scores, last_round):
		target = min(20, self.end_score - scores[self.index])
		if last_round:
			target = max(scores) - scores[self.index] + 1
		while sum(self.current_throws) < target:
			yield True
		yield False

class LastRound(Bot):
	def make_throw(self, scores, last_round):
		while sum(self.current_throws) < 15 and not last_round and scores[self.index] + sum(self.current_throws) < self.end_score:
			yield True
		while max(scores) > scores[self.index] + sum(self.current_throws):
			yield True
		yield False

class ThrowThriceBot(Bot):

	def make_throw(self, scores, last_round):
		yield True
		yield True
		yield False 

class NotTooFarBehindBot(Bot):
	def make_throw(self, scores, last_round):
		while True:
			current_score = scores[self.index] + sum(self.current_throws)
			number_of_bots_ahead = sum(1 for x in scores if x > current_score)
			if number_of_bots_ahead > 1:
				yield True
				continue
			if number_of_bots_ahead != 0 and last_round:
				yield True
				continue
			break
		yield False

class BringMyOwn_dice(Bot):

	def __init__(self, *args):
		# import random as rnd
		self.die = lambda: random.randint(1,6)
		super().__init__(*args)

	def make_throw(self, scores, last_round):

		nfaces = self.die() + self.die()

		s = scores[self.index]
		max_scores = max(scores)

		for _ in range(nfaces):
			if s + sum(self.current_throws) > self.end_score-1:
				break
			yield True

		yield False

class FooBot(Bot):
	def make_throw(self, scores, last_round):
		max_score = max(scores)

		while True:
			round_score = sum(self.current_throws)
			my_score = scores[self.index] + round_score

			if last_round:
				if my_score >= max_score:
					break
			else:
				if my_score >= self.end_score or round_score >= 16:
					break

			yield True

		yield False

class Roll6Timesv2(Bot):
	def make_throw(self, scores, last_round):

		if not last_round:
			i = 0
			maximum=6
			while ((i<maximum) and sum(self.current_throws)+scores[self.index]<=self.end_score ):
				yield True
				i=i+1

		if last_round:
			while scores[self.index] + sum(self.current_throws) < max(scores):
				yield True
		yield False

class Hesitate(Bot):
	def make_throw(self, scores, last_round):
		myscore = scores[self.index]
		if last_round:
			target = max(scores)+1
		elif myscore==0:
			target = 17
		else:
			target = self.end_score-5
		while myscore+sum(self.current_throws) < target:
			yield True
		yield False

class QuotaBot(Bot):
	def __init__(self, *args):
		super().__init__(*args)
		self.quota = 20
		self.minquota = 15
		self.maxquota = 35

	def make_throw(self, scores, last_round):
		# Reduce quota if ahead, increase if behind
		mean = sum(scores) / len(scores)
		own_score = scores[self.index]

		if own_score < mean - 5:
			self.quota += 1.5
		if own_score > mean + 5:
			self.quota -= 1.5

		self.quota = max(min(self.quota, self.maxquota), self.minquota)

		if last_round:
			self.quota = max(scores) - own_score + 1

		while sum(self.current_throws) < self.quota:
			yield True

		yield False

class Alpha(Bot):
	def make_throw(self, scores, last_round):
		# Throw until we're the best.
		while scores[self.index] + sum(self.current_throws) <= max(scores):
			yield True

		# Throw once more to assert dominance.
		yield True
		yield False

class GoBigEarly(Bot):
	def make_throw(self, scores, last_round):
		yield True  # always do a 2nd roll
		while scores[self.index] + sum(self.current_throws) < 25:
			yield True
		yield False

class ExpectationsBot(Bot):

	def make_throw(self, scores, last_round):
		#Positive average gain is 2.5, is the chance of loss greater than that?
		costOf6 = sum(self.current_throws) if scores[self.index] + sum(self.current_throws) < self.end_score  else scores[self.index] + sum(self.current_throws)
		while 2.5 > (costOf6 / 6.0):
			yield True
			costOf6 = sum(self.current_throws) if scores[self.index] + sum(self.current_throws) < self.end_score  else scores[self.index] + sum(self.current_throws)
		yield False

class StopBot(Bot):
	def make_throw(self, scores, last_round):
		yield False

class StepBot(Bot):
	def __init__(self, *args):
		super().__init__(*args)
		self.cycles = 0
		self.steps = 7
		self.smallTarget = 15
		self.bigTarget = 20
		self.rush = True

	def make_throw(self, scores, last_round):
		# Stacks upon stacks upon stacks
		self.bigTarget += 1 - (1 if self.steps % 3 == 0 else 0)
		self.cycles += 1
		self.steps += 1
		if self.cycles <=3:
			self.smallTarget += 1
		# If you didn't start the last round, panic ensues
		if last_round:
			# Rack up points just in case
			# while scores[self.index] + sum(self.current_throws) <= 35:
			#	 yield True
			# Keep going until we're not behind
			while max(scores) >= scores[self.index] + sum(self.current_throws):
				yield True
		else:
			# Hope for big points when low, don't bite more than you can chew when high
			target = self.bigTarget if scores[self.index] < 12 else self.bigTarget if self.cycles <=4 else self.smallTarget
			currentStep = 1
			while currentStep <= self.steps:
				if sum(self.current_throws) > target:
					break;
				yield True
				# After throw, if we get to 40 then rush (worst case we'll get drawn back)
				if scores[self.index] + sum(self.current_throws) > 40 and self.rush:
					currentStep = 1
					self.steps = 2
					self.rush = False
					target = self.smallTarget - 5
				currentStep += 1
		yield False

class TakeFive(Bot):
	def make_throw(self, scores, last_round):
		# Throw until we hit a 5.
		while self.current_throws[-1] != 5:
			# Don't get greedy.
			if scores[self.index] + sum(self.current_throws) >= self.end_score:
				break
			yield True

		# Go for the win on the last round.
		if last_round:
			while scores[self.index] + sum(self.current_throws) <= max(scores):
				yield True

		yield False

class GoHomeBot(Bot):
	def make_throw(self, scores, last_round):
		while scores[self.index] + sum(self.current_throws) < self.end_score:
			yield True
		yield False

class BinaryBot(Bot):

	def make_throw(self, scores, last_round):
		target = (self.end_score + scores[self.index]) / 2
		if last_round:
			target = max(scores)

		while scores[self.index] + sum(self.current_throws) < target:
			yield True

		yield False

class LeadBy5Bot(Bot):
	def make_throw(self, scores, last_round):
		while True:
			current_score = scores[self.index] + sum(self.current_throws)
			score_to_beat = max(scores) + 5
			if current_score >= score_to_beat:
				break
			yield True
		yield False

class EnsureLead(Bot):

	def make_throw(self, scores, last_round):
		otherScores = scores[self.index+1:] + scores[:self.index]
		maxOtherScore = max(otherScores)
		maxOthersToCome = 0
		for i in otherScores:
			if (i >= self.end_score): break
			else: maxOthersToCome = max(maxOthersToCome, i)
		while True:
			currentScore = sum(self.current_throws)
			totalScore = scores[self.index] + currentScore
			if not last_round:
				if totalScore >= self.end_score:
					if totalScore < maxOtherScore + 10:
						yield True
					else:
						yield False
				elif currentScore < 20:
					yield True
				else:
					yield False
			else:
				if totalScore < maxOtherScore + 1:
					yield True
				elif totalScore < maxOthersToCome + 10:
					yield True
				else:
					yield False

class PointsAreForNerdsBot(Bot):
	def make_throw(self, scores, last_round):
		while True:
			yield True

class OneInFiveBot(Bot):
	def make_throw(self, scores, last_round):
		while random.randint(1,5) < 5:
			yield True
		yield False

class FortyTeen(Bot):
	def make_throw(self, scores, last_round):
		if last_round:
			max_projected_score = max([score+14 if score<self.end_score else score for score in scores])
			target = max_projected_score - scores[self.index]
		else:
			target = 14

		while sum(self.current_throws) < target:
			yield True
		yield False

class BlessRNG(Bot):
	def make_throw(self, scores, last_round):
		if random.randint(1,2) == 1 :
			yield True
		yield False

class Chaser(Bot):
	def make_throw(self, scores, last_round):
		while max(scores) > (scores[self.index] + sum(self.current_throws)):
			yield True
		while last_round and (scores[self.index] + sum(self.current_throws)) < 44:
			yield True
		while self.not_thrown_firce() and sum(self.current_throws, scores[self.index]) < 44:
			yield True
		yield False

	def not_thrown_firce(self):
		return len(self.current_throws) < 4

class SlowStart(Bot):
	def __init__(self, *args):
		super().__init__(*args)
		self.completeLastRound = False
		self.nor = 1
		self.threshold = 8

	def updateValues(self):
		if self.completeLastRound:
			if self.nor < self.threshold:
				self.nor *= 2
			else:
				self.nor += 1
		else:
			self.threshold = self.nor // 2
			self.nor = 1


	def make_throw(self, scores, last_round):

		self.updateValues()
		self.completeLastRound = False

		i = 1
		while i < self.nor:
			yield True
			i += 1

		self.completeLastRound = True
		yield False

class FutureBot(Bot):
	def make_throw(self, scores, last_round):
		while (random.randint(1,6) != 6) and (random.randint(1,6) != 6):
			current_score = scores[self.index] + sum(self.current_throws)
			if current_score > (self.end_score+5):
				break
			yield True
		yield False

class OneStepAheadBot(Bot):
	def make_throw(self, scores, last_round):
		while random.randint(1,6) != 6:
			current_score = scores[self.index] + sum(self.current_throws)
			if current_score > (self.end_score+5):
				break
			yield True
		yield False

class FlipCoinRollDice(Bot):
	def make_throw(self, scores, last_round):
		while random.randint(1,2) == 2:
			throws = random.randint(1,6) != 6
			x = 0
			while x < throws:
				x = x + 1
				yield True
		yield False