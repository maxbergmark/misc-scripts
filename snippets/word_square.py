import urllib.request
words = set(
	urllib.request.urlopen(
		"http://www.mit.edu/~ecprice/wordlist.10000"
	).read()
	.decode()
	.split()
)

square = [
	"wcbqlqhtwgayj",
	"bfhtgaycxqqtx",
	"vrhczdtgayxfq",
	"cvgaykcxwpfhd",
	"tbqlqgayltvtg",
	"sdwhfxcqjbqla",
	"gflqtbgayrdhy",
	"abfjvqhlvqfcb",
	"ytqhncwzshdtx",
	"zvgayxljtlgmh",
	"ljcqvjhtfqapq",
	"pgaytpcqxvyjb"
]

def check(i, j, word, description):
	if len(word) > 2 and word in words:
		print(
			"row: %2d\tcol: %2d\tword: %s\t%s"
			% (i+1, j+1, word, description)
		)

for i in range(len(square)):
	for j in range(len(square[i])):
		for l in range(j, len(square[i])+1):
			word_h = square[i][j:l]
			check(i, j, word_h, "horizontal")

		word_v = ""
		for l in range(i, len(square)):
			word_v += square[l][j]
			check(i, j, word_v, "vertical")

		word_d = ""
		for l in range(min(len(square)-i, len(square[0])-j)):
			word_d += square[i+l][j+l]
			check(i, j, word_d, "diagonal down-right")

		word_dx = ""
		for l in range(min(len(square)-i, j)):
			word_dx += square[i+l][j-l]
			check(i, j, word_dx, "diagonal down-left")

		word_dx2 = ""
		for l in range(min(i, len(square[0])-j)):
			word_dx2 += square[i-l][j+l]
			check(i, j, word_dx2, "diagonal up-right")

		word_dx3 = ""
		for l in range(min(i, j)):
			word_dx3 += square[i-l][j-l]
			check(i, j, word_dx3, "diagonal up-left")