import random
import time
import math
import sys
from multiprocessing import Pool
from collections import defaultdict
# Importing all the bots
from forty_game_bots import *

# If you want to see what each bot decides, set this to true
# Should only be used with one thread and one game
DEBUG = False
# If your terminal supports ANSI, try setting this to true
ANSI = False

def print_str(x, y, string):
	print("\033["+str(y)+";"+str(x)+"H"+string, end = "", flush =   True)

class bcolors:
    WHITE = '\033[0m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

# Class for handling the game logic and relaying information to the bots
class Controller:

	def __init__(self, bots_per_game, games, bots, thread_id):
		"""Initiates all fields relevant to the simulation

		Keyword arguments:
		bots_per_game -- the number of bots that should be included in a game
		games -- the number of games that should be simulated
		bots -- a list of all available bot classes
		"""
		self.bots_per_game = bots_per_game
		self.games  = games
		self.bots = bots
		self.number_of_bots = len(self.bots)
		self.wins = {bot.__name__: 0 for bot in self.bots}
		self.played_games = {bot.__name__: 0 for bot in self.bots}
		self.end_score = 40
		self.thread_id = thread_id
		self.max_rounds = 200
		self.timed_out_games = 0
		self.tied_games = 0

	# Returns a fair dice throw
	def throw_die(self):
		return random.randint(1,6)
	# Print the current game number without newline
	def print_progress(self, progress):
		length = 50
		filled = int(progress*length)
		fill = "="*filled
		space = " "*(length-filled)
		perc = int(100*progress)
		if ANSI:
			col = [
				bcolors.RED, 
				bcolors.YELLOW, 
				bcolors.WHITE, 
				bcolors.BLUE, 
				bcolors.GREEN
			][int(progress*4)]

			end = bcolors.ENDC
			print_str(5, 8 + self.thread_id, 
				"\t%s[%s%s] %3d%%%s" % (col, fill, space, perc, end)
			)
		else:
			print(
				"\r\t[%s%s] %3d%%" % (fill, space, perc),
				flush = True, 
				end = ""
			)

	# Handles selecting bots for each game, and counting how many times
	# each bot has participated in a game
	def simulate_games(self):
		for game in range(self.games):
			if game % (self.games // 100) == 0 and not DEBUG:
				if self.thread_id == 0 or ANSI:
					progress = (game+1) / self.games
					self.print_progress(progress)
			game_bot_indices = random.sample(
				range(self.number_of_bots), 
				self.bots_per_game
			)

			game_bots = [None for _ in range(self.bots_per_game)]
			for i, bot_index in enumerate(game_bot_indices):
				self.played_games[self.bots[bot_index].__name__] += 1
				game_bots[i] = self.bots[bot_index](i, self.end_score)

			self.play(game_bots)
		if not DEBUG and (ANSI or self.thread_id == 0):
			self.print_progress(1)

		self.collect_results()

	def play(self, game_bots):
		"""Simulates a single game between the bots present in game_bots

		Keyword arguments:
		game_bots -- A list of instantiated bot objects for the game
		"""
		last_round = False
		last_round_initiator = -1
		round_number = 0
		game_scores = [0 for _ in range(self.bots_per_game)]

		# continue until one bot has reached end_score points
		while not last_round:
			for index, bot in enumerate(game_bots):

				self.single_bot(index, bot, game_scores, last_round)

				if game_scores[index] >= self.end_score:
					last_round = True
					last_round_initiator = index
			round_number += 1

			# maximum of 200 rounds per game
			if round_number > self.max_rounds - 1:
				last_round = True
				self.timed_out_games += 1
				# this ensures that everyone gets their last turn
				last_round_initiator = self.bots_per_game

		# make sure that all bots get their last round
		for index, bot in enumerate(game_bots[:last_round_initiator]):
			self.single_bot(index, bot, game_scores, last_round)

		# calculate which bots have the highest score
		max_score = max(game_scores)
		nr_of_winners = 0
		for i in range(self.bots_per_game):
			if game_scores[i] == max_score:
				nr_of_winners += 1
				self.wins[game_bots[i].__class__.__name__] += 1
		if nr_of_winners > 1:
			self.tied_games += 1

	def single_bot(self, index, bot, game_scores, last_round):
		"""Simulates a single round for one bot

		Keyword arguments:
		index -- The player index of the bot (e.g. 0 if the bot goes first)
		bot -- The bot object about to be simulated
		game_scores -- A list of ints containing the scores of all players
		last_round -- Boolean describing whether it is currently the last round
		"""

		current_throws = [self.throw_die()]
		if current_throws[-1] != 6:

			bot.update_state(current_throws[:])
			for throw in bot.make_throw(game_scores, last_round):
				# send the last die cast to the bot
				if not throw:
					break
				current_throws.append(self.throw_die())
				if current_throws[-1] == 6:
					break
				bot.update_state(current_throws[:])

		if current_throws[-1] == 6:
			# reset total score if running total is above end_score
			if game_scores[index] + sum(current_throws) - 6 >= self.end_score:
				game_scores[index] = 0
		else:
			# add to total score if no 6 is cast
			game_scores[index] += sum(current_throws)
		if DEBUG:
			desc = "%d: Bot %24s plays %40s with " + \
			"scores %20s and last round == %5s"
			print(desc % (index, bot.__class__.__name__, 
				current_throws, game_scores, last_round))


	# Collects all stats for the thread, so they can be summed in the main thread
	def collect_results(self):
		self.bot_stats = {
			bot.__name__: [
				self.wins[bot.__name__],
				self.played_games[bot.__name__]
			]
		for bot in self.bots}


# Print the high score after the simulation
def print_results(total_bot_stats, total_game_stats, elapsed_time):
	# Find the name of each bot, the number of wins, the number
	# of played games, and the win percentage
	wins = defaultdict(int)
	played_games = defaultdict(int)
	bots = set()
	timed_out_games = sum(s[0] for s in total_game_stats)
	tied_games = sum(s[1] for s in total_game_stats)

	for thread in total_bot_stats:
		for bot, stats in thread.items():
			wins[bot] += stats[0]
			played_games[bot] += stats[1]
			bots.add(bot)

	bot_stats = [[
		bot, 
		wins[bot],
		played_games[bot],
		0
	] for bot in bots]

	for i, bot in enumerate(bot_stats):
		if bot[2] > 0:
			bot[3] = 100 * bot[1] / bot[2]
		bot_stats[i] = tuple(bot)


	# Sort the bots by their winning percentage
	sorted_scores = sorted(bot_stats, key=lambda x: x[3], reverse=True)
	# Find the longest class name for any bot
	max_len = max([len(b[0]) for b in bot_stats])

	# Print the highscore list
	if ANSI:
		print_str(0, 9 + threads, "")
	else:
		print("\n")

	print("\tSimulation completed in %.1f seconds" % elapsed_time)
	print("\t%d games were ties between two or more bots" % tied_games)
	print("\t%d games ran until max_rounds\n" % timed_out_games)


	print("\t%s %s%5s (%8s/%8s)" 
		% ("Bot", " "*(max_len-2), "Win %", "Wins", "Played"))
	for bot, wins, played, score in sorted_scores:
		space_fill = " "*(max_len-len(bot)+1)
		format_arguments = (bot, space_fill, score, wins, played)
		print("\t%s:%s%5.1f (%8d/%8d)" % format_arguments)
	print()

def run_simulation(thread_id, bots_per_game, games_per_thread, bots):
	try:
		controller = Controller(bots_per_game, 
			games_per_thread, bots, thread_id)
		controller.simulate_games()
		controller_stats = (
			controller.timed_out_games,
			controller.tied_games
		)
		return (controller.bot_stats, controller_stats)
	except KeyboardInterrupt:
		return {}


# Prints the help for the script
def print_help():
	print("\nThis is the controller for the PPCG KotH challenge " + \
		"'A game of dice, but avoid number 6'")
	print("For any question, send a message to maxb\n")
	print("Usage: python %s [OPTIONS]" % sys.argv[0])
	print("\n  -n\t\tthe number of games to simluate")
	print("  -b\t\tthe number of bots per round")
	print("  -t\t\tthe number of threads")
	print("  -A\t\tRun in ANSI mode, with prettier printing")
	print("  -h\t--help\tshow this help\n")

if __name__ == "__main__":

	bots = get_all_bots()
	games = 10000
	bots_per_game = 8
	threads = 4

	for i, arg in enumerate(sys.argv):
		if arg == "-n" and len(sys.argv) > i+1 and sys.argv[i+1].isdigit():
			games = int(sys.argv[i+1])
		if arg == "-b" and len(sys.argv) > i+1 and sys.argv[i+1].isdigit():
			bots_per_game = int(sys.argv[i+1])
		if arg == "-t" and len(sys.argv) > i+1 and sys.argv[i+1].isdigit():
			threads = int(sys.argv[i+1])
		if arg == "-A":
			ANSI = True
		if arg == "-h" or arg == "--help":
			print_help()
			quit()
	if ANSI:
		print(chr(27) + "[2J", flush =  True)
		print_str(1,3,"")
	else:
		print()

	if bots_per_game > len(bots):
		bots_per_game = len(bots)
	if bots_per_game < 2:
		print("\tAt least 2 bots per game is needed")
		bots_per_game = 2
	if games <= 0:
		print("\tAt least 1 game is needed")
		games = 1
	if threads <= 0:
		print("\tAt least 1 thread is needed")
		threads = 1

	games_per_thread = math.ceil(games / threads)

	print("\tStarting simulation with %d bots" % len(bots))
	sim_str = "\tSimulating %d games with %d bots per game"
	print(sim_str % (games, bots_per_game))
	print("\tRunning simulation on %d threads" % threads)
	if len(sys.argv) == 1:
		print("\tFor help running the script, use the -h flag")
	print()
	with Pool(threads) as pool:
		t0 = time.time()
		results = pool.starmap(
			run_simulation, 
			[(i, bots_per_game, games_per_thread, bots) for i in range(threads)]
		)
		t1 = time.time()
		if not DEBUG:
			total_bot_stats = [r[0] for r in results]
			total_game_stats = [r[1] for r in results]
			print_results(total_bot_stats, total_game_stats, t1-t0)