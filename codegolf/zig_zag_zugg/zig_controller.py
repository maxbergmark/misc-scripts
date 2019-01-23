#!/usr/bin/env python3
import re
import json
import math
import random
import requests
import sys
import time
from numpy import cumsum

from collections import defaultdict
from html import unescape
from lxml import html
from multiprocessing import Pool
from os import path, rename, remove
from sys import stderr
from time import strftime

# If you want to see what each bot decides, set this to true
# Should only be used with one thread and one game
DEBUG = False
# If your terminal supports ANSI, try setting this to true
ANSI = False
# File to keep base class and own bots
OWN_FILE = 'zig_zag_bots.py'
# File where to store the downloaded bots
AUTO_FILE = 'auto_bots.py'
# If you want to use up all your quota & re-download all bots
DOWNLOAD = False
# If you want to ignore a specific user's bots (eg. your own bots): add to list
IGNORE = []
# The API-request to get all the bots
URL = "https://api.stackexchange.com/2.2/questions/177765/answers?page=%s&pagesize=100&order=desc&sort=creation&site=codegolf&filter=!bLf7Wx_BfZlJ7X"


def print_str(x, y, string):
	print("\033["+str(y)+";"+str(x)+"H"+string, end = "", flush = True)

class bcolors:
	WHITE = '\033[0m'
	GREEN = '\033[92m'
	BLUE = '\033[94m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	ENDC = '\033[0m'

# Class for handling the game logic and relaying information to the bots
class Controller:

	def __init__(self, games, bots_per_game, bots, thread_id):
		"""Initiates all fields relevant to the simulation

		Keyword arguments:
		games -- the number of games that should be simulated
		bots -- a list of all available bot classes
		"""
		self.games = games
		self.bots = [bot() for bot in bots]
		self.bots_per_game = bots_per_game
		self.number_of_bots = len(self.bots)
		self.wins = defaultdict(int)
		self.played_games = defaultdict(int)
		self.bot_timings = defaultdict(float)
		self.thread_id = thread_id
		self.max_rounds = 100
		self.timed_out_games = 0
		self.total_rounds = 0
		self.highest_round = 0

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
			if self.games > 100:
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
				self.played_games[self.bots[bot_index].__class__.__name__] += 1
				game_bots[i] = self.bots[bot_index]

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
		round_number = 0
		previous_announcements = []
		previous_choices = []
		announcements = [-1 for _ in range(self.bots_per_game)]
		decisions = [-1 for _ in range(self.bots_per_game)]
		in_game = [True for _ in range(self.bots_per_game)]
		play_until_one = False

		bot_names = [bot.__class__.__name__ for bot in game_bots]
		for i, bot in enumerate(game_bots):
			bot.reset(i, bot_names)

		# continue until one bot has reached end_score points
		while not last_round:
			last_round = (round_number == self.max_rounds)
			for index, bot in enumerate(game_bots):
				if in_game[index]:
					t0 = time.clock()
					announcements[index] = bot.announce(
						previous_announcements, 
						previous_choices, 
						in_game,
						last_round
					)
					t1 = time.clock()
					self.bot_timings[bot.__class__.__name__] += t1-t0
				else:
					announcements[index] = None
			for index, bot in enumerate(game_bots):
				if in_game[index]:
					t0 = time.clock()
					decisions[index] = bot.decide(announcements)
					t1 = time.clock()
					self.bot_timings[bot.__class__.__name__] += t1-t0
				else:
					decisions[index] = None

			for index, bot in enumerate(game_bots):
				if in_game[index]:
					t0 = time.clock()
					bot.recap(decisions)
					t1 = time.clock()
					self.bot_timings[bot.__class__.__name__] += t1-t0
				else:
					decisions[index] = None

			true_count = decisions.count(True)
			false_count = decisions.count(False)
			if play_until_one:
				if true_count < false_count and true_count > 0:
					for i, decision in enumerate(decisions):
						if decision == False:
							in_game[i] = False

				if true_count > false_count and false_count > 0:
					for i, decision in enumerate(decisions):
						if decision == True:
							in_game[i] = False

				if sum(in_game) == 1:
					winner = game_bots[decisions.index(True)]
					self.wins[winner.__class__.__name__] += 1
					# print("winner found")
					break
			else:
				if true_count < false_count:
					for i, bot in enumerate(game_bots):
						if decisions[i] == True:
							self.wins[bot.__class__.__name__] += 1

				if true_count > false_count:
					for i, bot in enumerate(game_bots):
						if decisions[i] == False:
							self.wins[bot.__class__.__name__] += 1
				break

			round_number += 1

		self.total_rounds += round_number
		self.highest_round = max(self.highest_round, round_number)


	# Collects all stats for the thread, so they can be summed up later
	def collect_results(self):
		self.bot_stats = {
			bot.__class__.__name__: [
				self.wins[bot.__class__.__name__],
				self.played_games[bot.__class__.__name__]
			]
		for bot in self.bots}


# 
def print_results(total_bot_stats, total_game_stats, elapsed_time):
	"""Print the high score after the simulation

	Keyword arguments:
	total_bot_stats -- A list containing the winning stats for each thread
	total_game_stats -- A list containing controller stats for each thread
	elapsed_time -- The number of seconds that it took to run the simulation
	"""

	# Find the name of each bot, the number of wins, the number
	# of played games, and the win percentage
	wins = defaultdict(int)
	played_games = defaultdict(int)
	# highscores = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
	bots = set()
	timed_out_games = sum(s[0] for s in total_game_stats)
	total_games = sum(s[1] for s in total_game_stats)
	total_rounds = sum(s[3] for s in total_game_stats)
	highest_round = max(s[4] for s in total_game_stats)
	average_rounds = total_rounds / total_games
	winning_scores = defaultdict(int)
	bot_timings = defaultdict(float)
	
	for thread in total_bot_stats:
		for bot, stats in thread.items():
			wins[bot] += stats[0]
			played_games[bot] += stats[1]

			# highscores[bot][0] = max(highscores[bot][0], stats[2][0])	   
			# for i in range(1, 6):
				# highscores[bot][i] += stats[2][i]
			bots.add(bot)
	
	for bot in bots:
		bot_timings[bot] += sum(s[2][bot] for s in total_game_stats)

	bot_stats = [[bot, wins[bot], played_games[bot], 0] for bot in bots]

	for i, bot in enumerate(bot_stats):
		bot[3] = 100 * bot[1] / bot[2] if bot[2] > 0 else 0
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


	sim_msg = "\tSimulation or %d games between %d bots " + \
		"completed in %.1f seconds"
	print(sim_msg % (total_games, len(bots), elapsed_time))
	print("\tEach game lasted for an average of %.2f rounds" % average_rounds)
	print("\t%d games ran until the round limit, highest round was %d\n"
		% (timed_out_games, highest_round))

	print_bot_stats(sorted_scores, max_len)
	print_time_stats(bot_timings, max_len)


def print_bot_stats(sorted_scores, max_len):
	"""Print the stats for the bots

	Keyword arguments:
	sorted_scores -- A list containing the bots in sorted order
	max_len -- The maximum name length for all bots
	highscores -- A dict with additional stats for each bot
	"""
	delimiter_format = "\t+%s%s+%s+%s+%s+"
	delimiter_args = ("-"*(max_len), "", "-"*4, "-"*8, 
		"-"*8)
	delimiter_str = delimiter_format % delimiter_args
	print(delimiter_str)
	print("\t|%s%s|%4s|%8s|%8s|" 
		% ("Bot", " "*(max_len-3), "Win%", "Wins", 
			"Played"))
	print(delimiter_str)

	for bot, wins, played, score in sorted_scores:

		space_fill = " "*(max_len-len(bot))
		format_str = "\t|%s%s|%4.1f|%8d|%8d|"
		format_arguments = (bot, space_fill, score, wins, played)
		print(format_str % format_arguments)

	print(delimiter_str)
	print()

def print_time_stats(bot_timings, max_len):
	"""Print the execution time for all bots

	Keyword arguments:
	bot_timings -- A dict containing information about timings for each bot
	max_len -- The maximum name length for all bots
	"""
	total_time = sum(bot_timings.values())
	sorted_times = sorted(bot_timings.items(), 
		key=lambda x: x[1], reverse = True)

	delimiter_format = "\t+%s+%s+%s+"
	delimiter_args = ("-"*(max_len), "-"*7, "-"*5)
	delimiter_str = delimiter_format % delimiter_args
	print(delimiter_str)

	print("\t|%s%s|%7s|%5s|" % ("Bot", " "*(max_len-3), "Time", "Time%"))
	print(delimiter_str)
	for bot, bot_time in sorted_times:
		space_fill = " "*(max_len-len(bot))
		perc = 100 * bot_time / total_time
		print("\t|%s%s|%7.2f|%5.1f|" % (bot, space_fill, bot_time, perc))
	print(delimiter_str)
	print() 


def run_simulation(thread_id, games_per_thread, bots_per_game, bots):
	"""Used by multithreading to run the simulation in parallel

	Keyword arguments:
	thread_id -- A unique identifier for each thread, starting at 0
	games_per_thread -- The number of games to be simulated
	bots -- A list of all bot classes available
	"""
	try:
		controller = Controller(games_per_thread, bots_per_game, bots, thread_id)
		controller.simulate_games()
		controller_stats = (
			controller.timed_out_games,
#			controller.tied_games,
			controller.games,
			controller.bot_timings,
			controller.total_rounds,
			controller.highest_round
#			controller.winning_scores
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
	print("  -d\t--download\tdownload all bots from codegolf.SE")
	print("  -A\t--ansi\trun in ANSI mode, with prettier printing")
	print("  -D\t--debug\trun in debug mode. Sets to 1 thread, 1 game")
	print("  -h\t--help\tshow this help\n")

# Make a stack-API request for the n-th page
def req(n):
	req = requests.get(URL % n)
	req.raise_for_status()
	return req.json()

# Pull all the answers via the stack-API
def get_answers():
	n = 1
	api_ans = req(n)
	answers = api_ans['items']
	while api_ans['has_more']:
		n += 1
		if api_ans['quota_remaining']:
			api_ans = req(n)
			answers += api_ans['items']
		else:
			break

	m, r = api_ans['quota_max'], api_ans['quota_remaining']
	if 0.1 * m > r:
		print(" > [WARN]: only %s/%s API-requests remaining!" % (r,m), file=stderr)

	return answers


def download_players():
	players = {}

	for ans in get_answers():
		name = unescape(ans['owner']['display_name'])
		bots = []

		root = html.fromstring('<body>%s</body>' % ans['body'])
		for el in root.findall('.//code'):
			code = el.text
			if re.search(r'^class \w+\(\w*Bot\):.*$', code, flags=re.MULTILINE):
				bots.append(code)

		if not bots:
			print(" > [WARN] user '%s': couldn't locate any bots" % name, file=stderr)
		elif name in players:
			players[name] += bots
		else:
			players[name] = bots

	return players


# Download all bots from codegolf.stackexchange.com
def download_bots():
	print('pulling bots from the interwebs..', file=stderr)
	try:
		players = download_players()
	except Exception as ex:
		print('FAILED: (%s)' % ex, file=stderr)
		exit(1)

	if path.isfile(AUTO_FILE):
		print(' > move: %s -> %s.old' % (AUTO_FILE,AUTO_FILE), file=stderr)
		if path.exists('%s.old' % AUTO_FILE):
			remove('%s.old' % AUTO_FILE)
		rename(AUTO_FILE, '%s.old' % AUTO_FILE)

	print(' > writing players to %s' % AUTO_FILE, file=stderr)
	f = open(AUTO_FILE, 'w+', encoding='utf8')
	f.write('# -*- coding: utf-8 -*- \n')
	f.write('# Bots downloaded from https://codegolf.stackexchange.com/questions/177765 @ %s\n\n' % strftime('%F %H:%M:%S'))
	with open(OWN_FILE, 'r') as bfile:
		f.write(bfile.read()+'\n\n\n# Auto-pulled bots:\n\n')
	for usr in players:
		if usr not in IGNORE:
			for bot in players[usr]:
				f.write('# User: %s\n' % usr)
				f.write(bot+'\n\n')
	f.close()

	print('OK: pulled %s bots' % sum(len(bs) for bs in players.values()))


if __name__ == "__main__":

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
		if arg == "-d" or arg == "--download":
			DOWNLOAD = True
		if arg == "-A" or arg == "--ansi":
			ANSI = True
		if arg == "-D" or arg == "--debug":
			DEBUG = True
		if arg == "-h" or arg == "--help":
			print_help()
			quit()
	if ANSI:
		print(chr(27) + "[2J", flush =  True)
		print_str(1,3,"")
	else:
		print()

	if DOWNLOAD:
		download_bots()
		exit() # Before running other's code, you might want to inspect it..

	if path.isfile(AUTO_FILE):
		exec('from %s import *' % AUTO_FILE[:-3])
	else:
		exec('from %s import *' % OWN_FILE[:-3])

	bots = get_all_bots()

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
	if DEBUG:
		print("\tRunning in debug mode, with 1 thread and 1 game")
		threads = 1
		games = 1

	games_per_thread = math.ceil(games / threads)

	print("\tStarting simulation with %d bots" % len(bots))
	sim_str = "\tSimulating %d games with %d bots per game"
	print(sim_str % (games, bots_per_game))
	print("\tRunning simulation on %d threads" % threads)
	if len(sys.argv) == 1:
		print("\tFor help running the script, use the -h flag")
	print()

	if threads == 1:
		t0 = time.time()
		results = [run_simulation(0, games_per_thread, bots_per_game, bots)]
		t1 = time.time()
		if not DEBUG:
			total_bot_stats = [r[0] for r in results]
			total_game_stats = [r[1] for r in results]
			print_results(total_bot_stats, total_game_stats, t1-t0)
	else:
		with Pool(threads) as pool:
			t0 = time.time()
			results = pool.starmap(
				run_simulation, 
				[(i, games_per_thread, bots_per_game, bots) for i in range(threads)]
			)
			t1 = time.time()
			if not DEBUG:
				total_bot_stats = [r[0] for r in results]
				total_game_stats = [r[1] for r in results]
				print_results(total_bot_stats, total_game_stats, t1-t0)