import numpy as np
import time

def get_main_diag(index, n):
	min_row = max(0, index[0]-index[1])
	min_col = max(0, index[1]-index[0])
	n_diag = n - max(min_row, min_col)		
	main_diag = np.vstack((
		np.arange(min_row, min_row+n_diag), 
		np.arange(min_col, min_col+n_diag)
	)).T
	return main_diag

def get_anti_main_diag(index, n):
	anti_min_row = min(n-1, index[0]+index[1])
	anti_min_col = index[1] - (anti_min_row - index[0])
	anti_n_diag = n - max(n-1-anti_min_row, anti_min_col)		
	anti_main_diag = np.vstack((
		np.arange(anti_min_row, anti_min_row-anti_n_diag, -1), 
		np.arange(anti_min_col, anti_min_col+anti_n_diag)
	)).T
	return anti_main_diag

def setup_board(board, main_diag, anti_main_diag, index):
	board[main_diag[:,0], main_diag[:,1]] += 1
	board[anti_main_diag[:,0], anti_main_diag[:,1]] += 1
	board[index[0],:] += 1
	board[:,index[1]] += 1

def reset_board(board, main_diag, anti_main_diag, index):
	board[main_diag[:,0], main_diag[:,1]] -= 1
	board[anti_main_diag[:,0], anti_main_diag[:,1]] -= 1
	board[index[0],:] -= 1
	board[:,index[1]] -= 1

def iterate(board, queens, pos = []):
	if (queens == 0):
		show = 0*board
		for p in pos:
			show[p] = 1
		print(show)
		print()
		return True, show, 0

	elapsed = 0
	for index, value in np.ndenumerate(board):
		if value == 0:

			main_diag = get_main_diag(index, n)
			anti_main_diag = get_anti_main_diag(index, n)
			setup_board(board, main_diag, anti_main_diag, index)
			pos.append(index)
			# time.sleep(0.1)
			t0 = time.time()
			t1 = time.time()
			success, solution, elapsed_call = iterate(board, queens-1, pos)
			t2 = time.time()
			pos.pop()
			reset_board(board, main_diag, anti_main_diag, index)
			t3 = time.time()
			elapsed += elapsed_call + t3-t2 + t1-t0
			# print(" "*(4-queens) + str(elapsed) + "\t" + str(elapsed_call))


			if success:
				return success, solution, elapsed
			# break
	return False, None, elapsed

def print_board_compact(board, n):
	print("+" + "-"*n + "+")
	for row in range(n):
		s = "|"
		for col in range(n):
			s += " Q"[board[row, col]]
		s += "|"
		print(s)
	print("+" + "-"*n + "+")

def print_board(board, n):
	print("+" + "-+"*n)
	for row in range(n):
		s = "|"
		for col in range(n):
			s += " Q"[board[row, col]] + "|"
		print(s)
		print("+" + "-+"*n)
n = 6	
# board = np.arange(n*n).reshape((n, n))
board = np.zeros((n,n), dtype=np.int32)
success, solution, elapsed = iterate(board, n)
print_board(solution, n)
print(elapsed)
