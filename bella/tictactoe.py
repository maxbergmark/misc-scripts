def check_winner(board, player):
	n = len(board)
	for row in range(n):
		if board[row].count(player) == n:
			return True
	for col in range(n):
		temp = True
		for row  in range(n):
			if board[row][col] != player:
				temp = False
		if temp:
			return True

	temp = True
	for i in range(n):
		if board[i][i] != player:
			temp = False
	if temp:
		return True

	temp = True
	for i in range(n):
		if board[i][n-i-1] != player:
			temp = False
	if temp:
		return True

	return False

def print_board(board):
	n = len(board)
	# print('+-'*n + '+')
	for line in board:
		print('|'.join(line))
		# print('+-'*n + '+')

def prompt_move(board, player):
	n = len(board)
	while True:
		move = int(input("Make your move [1-%d]: " % (n**2,))) - 1
		row = move // n
		col = move % n
		if board[row][col] == ' ':
			break
		else:
			print("Impossible move")
	board[row][col] = player

n = 3
board = [[' ' for i in range(n)] for j in range(n)]
players = "XO"
num_players = len(players)
turn = 0

while True:
	player = players[turn % num_players]
	print_board(board)
	prompt_move(board, player)
	if check_winner(board, player):
		print_board(board)
		print("%s won the game!" % (player,))
		break
	turn += 1