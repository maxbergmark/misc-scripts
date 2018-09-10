import time
import numpy as np

x = [0]
def get_winner(board):
	for i in range(3):
		if board[i,:].sum() == 0: return 0
		if board[i,:].sum() == 3: return 1
		if board[:,i].sum() == 0: return 0
		if board[:,i].sum() == 3: return 1
	diag = board.diagonal()
	antidiag = board[:,::-1].diagonal()
	if diag.sum() == 0: return 0
	if diag.sum() == 3: return 1
	if antidiag.sum() == 0: return 0
	if antidiag.sum() == 3: return 1
	if board.sum() == 4: return 2
	return -1

def check_last_move(board, side, i, j):
	if board[i,:].sum() == 3*side: return True
	if board[:,j].sum() == 3*side: return True
	if (2*i+j)%3 == 0:
		diag = board.diagonal()
		if diag.sum() == 3*side: return True
	if i+j == 2:
		antidiag = board[:,::-1].diagonal()
		if antidiag.sum() == 3*side: return True
	return False

def check_immediate_win(board, side):
	for x in range(3):
		for y in range(3):
			if board[x,y] < 0:
				# print_board(board)
				board[x,y] = side
				if check_last_move(board, side, x, y):
					board[x,y] = -7
					return side
				board[x,y] = -7
	return -1

def make_move(board, side, i, j, level = 0):
	board[i,j] = side

	if check_last_move(board, side, i, j):
		board[i,j] = -7
		return side
	else:
		immediate = check_immediate_win(board, 1-side)
		if immediate >= 0:
			board[i,j] = -7
			return immediate
		best_move = 1-side
		for x in range(3):
			for y in range(3):
				if board[x,y] < 0:
					winner = make_move(board, 1-side, x, y, level+1)
					if best_move == 1-side and winner != 1-side:
					print('-'*level, (i, j), (x, y), 'OXD'[winner])

	board[i,j] = -7
	return -1


def get_best_move(board, side, level=0):
	best_move = (0,0,-2)
	for i in range(3):
		for j in range(3):
			if board[i,j] < 0:
				board[i,j] = side
				print_board(board)
				board[i,j] = -7
				winner = make_move(board, side, i, j, 0)
				print("Winner for move:",i,j,'OX'[winner] if winner>=0 else 'DRAW')
				print("\n-------\n")
				if winner == -1 or winner == side:
					if best_move[2] < 0:
						best_move = (i, j, winner)

	return best_move[:2]

def print_board(board):
	d = {-7:' ', 0:'O', 1:'X'}
	print('+-+-+-+\n|', end='')
	print('|\n+-+-+-+\n|'.join(['|'.join([d[i] for i in x]) for x in board]), end='')
	print('|\n+-+-+-+')

def prompt_move():
	move = str(input("Move: ")).split(' ')
	return tuple(int(i) for i in move)

board = np.ones((3,3)).astype(int)*-7
side = 0
board[0,0] = 0
# board[2,2] = 0
# board[1,1] = 0

board[0,1] = 1
# board[2,1] = 1

# print(make_move(board, side, 2,2))

# quit()
while get_winner(board) == -1:
	print("\n\n\n")
	print_board(board)
	if side == 0:
		# move = prompt_move()
		move = get_best_move(board, side)
	else:
		# move = get_best_move(board, side)
		move = prompt_move()

	print(move)
	board[move[0],move[1]] = side
	side = 1-side

winner = get_winner(board)
print_board(board)
print("Winner:", ["O","X","DRAW"][winner])

# move = get_best_move(board, side, 1)
# print(move)
# board[move[0],move[1]] = side
# print_board(board)
