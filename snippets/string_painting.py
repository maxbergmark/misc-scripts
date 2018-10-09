import numpy as np
import imageio
from matplotlib import pyplot as plt
from skimage.draw import line_aa
import random

def get_score(img, goal):
	diff = img - goal
	sqr_diff = (diff**2).sum()
	# neg_diff = diff[diff < 0].sum()
	return sqr_diff# - 10*neg_diff

def draw_random_line(img):
	case = random.random()
	last_row = img.shape[0]-1
	last_col = img.shape[1]-1
	if case < 1/6: # ul
		start_point = random.randint(0, last_col)
		end_point = random.randint(0, last_row)
		rows, cols, weights = line_aa(0, start_point, end_point, 0)
	elif case < 1/3: # ud
		start_point = random.randint(0, last_col)
		end_point = random.randint(0, last_col)
		rows, cols, weights = line_aa(0, start_point, last_row, end_point)
	elif case < 1/2: # ur
		start_point = random.randint(0, last_col)
		end_point = random.randint(0, last_row)
		rows, cols, weights = line_aa(0, start_point, end_point, last_col)
	elif case < 2/3: # ld
		start_point = random.randint(0, last_row)
		end_point = random.randint(0, last_col)
		rows, cols, weights = line_aa(start_point, 0, last_row, end_point)
	elif case < 5/6: # lr
		start_point = random.randint(0, last_row)
		end_point = random.randint(0, last_row)
		rows, cols, weights = line_aa(start_point, 0, end_point, last_col)

	else: # dr
		start_point = random.randint(0, last_col)
		end_point = random.randint(0, last_row)
		rows, cols, weights = line_aa(last_row, start_point, end_point, last_col)

	# print(last_row, last_col, case, start_point, end_point)
	img[rows, cols] = (1-weights/10)*img[rows, cols]

def iterate(img, target, score):
	copy = np.copy(img)
	for _ in range(10):
		draw_random_line(copy)
	copy_score = get_score(copy, target)
	# if copy_score < score:
	return copy, copy_score
	# return img, score

def generation(img, target, score):
	best_score = 2**32
	best_img = None
	for i in range(100):
		copy, copy_score = iterate(img, target, score)
		if copy_score < best_score:
			if copy_score < score:
				print("new best image", copy_score)
			best_score = copy_score
			best_img = copy
	if best_score < score:
		return best_img, best_score
	return img, score

target = imageio.imread("portrait_cropped_small.jpg")
# print(img.shape)
# im = np.mean(img, 2, dtype=np.int)


img = 0 * target + 255
average = np.mean(target)
print(average)
average_img = 0 * target + int(average)
average_score = get_score(average_img, target)

score = get_score(img, target)
for i in range(1000):
	# if i % 1000 == 0:
	print("Generation: %d (%d) (%.2f)" % (i, score, score/average_score))
	img, score = generation(img, target, score)

plt.imshow(img, "gray", vmin = 0, vmax = 255)
plt.show()


# ul
# ud
# ur

# ld
# lr

# dr