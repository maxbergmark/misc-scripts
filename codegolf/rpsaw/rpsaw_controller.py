import rpsaw_bots
import random

def play_rpsaw(bots, games, rounds_per_game):
	for game in range(games):
		game_bots = random.sample(bots, 2)


bots = rpsaw_bots.get_all_bots()
play_rpsaw(bots, 10, 9)