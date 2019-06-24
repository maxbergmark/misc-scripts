def print_board(board):
	for row in board:
		print("|".join(row))
board=[
	[" ", " ", " "],
	[" ", " ", " "],
	[" ", " ", " "]
]
print_board(board)
tic=int(input("vilken ruta vill du lägga in på?"))-1
print(tic)
row=tic//3
col=tic%3
board[row][col]="O"
print_board(board)