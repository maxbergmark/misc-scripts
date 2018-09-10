import matplotlib
# Remove this if your system doesn't have support for Qt5
matplotlib.use("Qt5agg")
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan, suppress=True)
from numpy.linalg import norm
import time
import random
from scipy.ndimage.filters import gaussian_filter
import sys
import warnings

# Generates a very simple path with two turns
def generate_easy_path(n):
	origin = np.array([0,0])
	path = np.zeros((n, 2))
	for i in range(n//2):
		path[i,:] = np.array([i,0])
	for i in range(n//2, 3*n//4):
		path[i,:] = np.array([n//2, i-n//2])
	for i in range(3*n//4, n):
		path[i,:] = np.array([i-n//4, i-n//2])
	return path

# Returns the cosine angle between two vectors
def get_vector_angle_cosine(v1, v2):
	return np.dot(v1,v2)/norm(v1)/norm(v2)

# Generates a modified brownian path stating from (0, 0)
# The path only changes direction with probability 0.3 for each step
# The direction can't change by more than 90 degrees to more closely
# resemble a human path
def generate_brownian_path(n):
	random.seed(1)
	pos = np.array([0,0])
	dirs = np.array([[1,0], [0,1], [-1,0], [0,-1], 
		[1,1], [1,-1], [-1,1], [-1,-1]])
	temp_dir = dirs[random.randint(0, 7),:]
	path = np.zeros((n, 2))
	for i in range(n):
		path[i,:] = pos
		pos += temp_dir
		if (random.random() < 0.3):
			new_dir = dirs[random.randint(0, 7),:]
			cos_angle = get_vector_angle_cosine(temp_dir, new_dir)
			while (cos_angle < 0):
				new_dir = dirs[random.randint(0, 7),:]
				cos_angle = get_vector_angle_cosine(temp_dir, new_dir)
			temp_dir = new_dir
	return path*.5

# Creates intermediary points for a path
# E.g. with a detail of 4 the input [[0,0], [1,0], [1,2]]
# becomes [[0,0], [.5,0], [1,0], [1,1], [1,2]]
# Interpolates for x and y independently
# Note that interpolated_path[::detail] == path
def interpolate_path(path, detail):
	part = 1/detail
	n = path.shape[0]
	interpolator = np.zeros(((n-1)*detail+1, n))
	interpolate_piece = np.zeros((detail, 2))

	for j in range(detail):
		interpolate_piece[j,0] = 1-j*part
		interpolate_piece[j,1] = j*part
	
	for i in range(n-1):
		interpolator[i*detail:(i+1)*detail,i:i+2] = interpolate_piece

	interpolator[-1,-1] = 1
	interpolated_path = np.dot(interpolator, path)
	return interpolated_path

# Applies a gaussian filter to x and y independently
# This makes the path smoother and more natural
# Try modifying the first sigma value to see the change in the curve
def filter_path(path, detail):
	smooth_path = gaussian_filter(path, sigma=(detail/1, 0))
	return smooth_path


# Calculates the area of a polygon, including the sign of the area
# Polygons that turn counter-clockwise have a positive area
# This follows the right-hand rule from linear algebra
def signed_polygon_area(p):
	correction = p[-1,0] * p[0,1] - p[-1,1]* p[0,0]
	main_area = np.dot(p[:-1,0], p[1:,1]) - np.dot(p[:-1,1], p[1:,0])
	return -0.5*(main_area + correction)


def signed_poly_area(x,y):
	return .5*(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


# This calculates all nodes of smooth_path which have turns in front
# of them. look_forward determines how far ahead to check for turns, 
# measured in number of nodes in the original path
# Try changing the threshold to see how the number of turns change
def calculate_turns_old(smooth_path, detail, look_forward = 3):
	look_forward *= detail
	threshold = 0.03 #arbitrary value, found empirically
	turns = []
	for i in range(0, len(smooth_path)-look_forward, 1):
		p0 = np.array([smooth_path[i,0], smooth_path[i,1]])
		p1 = np.array([smooth_path[i+look_forward,0], 
						smooth_path[i+look_forward,1]])
		segment_length = norm(p1-p0)

		temp_x = smooth_path[i:i+look_forward,0]
		temp_y = smooth_path[i:i+look_forward,1]
		turn_area = signed_poly_area(temp_x, temp_y)/segment_length**2
		if abs(turn_area) > threshold:
			turns.append([i, turn_area])

	return turns

def calculate_turns(smooth_path, detail, look_forward = 3):
	look_forward *= detail
	threshold = 0.03 #arbitrary value, found empirically
	turn_areas = [0 for i in range(len(smooth_path)-look_forward)]
	turns = []

	differences = smooth_path[look_forward:,:] - smooth_path[:-look_forward,:]
	distances = norm(differences, None, 1)**2

	for i in range(0, len(smooth_path)-look_forward, detail):
		p = smooth_path[i:i+look_forward]
		turn_area = signed_polygon_area(p)/distances[i]
		turn_areas[i] = turn_area
	
	over_threshold_left = np.abs(turn_areas[:-detail]) > threshold
	over_threshold_right = np.abs(turn_areas[detail:]) > threshold
	over_threshold = over_threshold_left | over_threshold_right

	for i in range(0, len(smooth_path)-look_forward-detail, detail):
		if over_threshold[i]:
			for j in range(i+1,i+detail):
				p = smooth_path[j:j+look_forward]
				turn_area = signed_polygon_area(p)/distances[j]
				turn_areas[j] = turn_area
	
	
	over_threshold = np.abs(turn_areas) > threshold
	for i in range(0, len(smooth_path)-look_forward):
		if (over_threshold[i]):
			turns.append([i, turn_areas[i]])

	return turns

# Finds all unique turns, by aggregating turns on adjacent nodes
# Returns the indices for each turn in the original path
def filter_turns(turns, detail):
	i = 0
	turn_nodes = []
	while i < len(turns):
		temp_index = i

		while (temp_index+1 < len(turns) and 
			turns[temp_index+1][0]-turns[temp_index][0] == 1 and
			turns[temp_index+1][1]*turns[temp_index][1] > 0):
			temp_index += 1

		average_index = (temp_index+i)//2
		# Arbitrary transform, makes "sharp" turns close to 1
		color_value = min(1, abs(turns[average_index][1])*4.5)**.5

		if turns[average_index][1] < 0:
			color = (color_value, 0, 0)
		else:
			color = (0, color_value, 0)

		turn_nodes.append([turns[average_index][0]//detail, color, 
			turns[i][0]//detail, 
			turns[temp_index][0]//detail])
		i = temp_index+1
	return turn_nodes

# Draws the arrows to the plot
def draw_path_arrows(path):
	ax = plt.axes()
	for i in range(len(path)-1):
		dx = (path[i+1,0]-path[i,0])
		dy = (path[i+1,1]-path[i,1])
		length = (dx*dx+dy*dy)**.5
		line_norm = (length-0.20)/length
		ax.arrow(path[i,0], path[i,1], dx*line_norm, dy*line_norm,
			head_width = 0.1, head_length = 0.15, fc='k')

# Draws the colored dots to the plot
def draw_turns(path, turns):
	offset = 2
	for turn in turns:
		plt.plot(path[turn[2]+offset,0], path[turn[2]+offset ,1], 'o', 
			color = turn[1], ms = 15)
		for i in range(turn[2], turn[3]):
			plt.plot([path[i+offset,0], path[i+offset+1,0]], 
				[path[i+offset,1], path[i+offset+1,1]], 
				linewidth = 5, color = turn[1])

# This is just a hack to be able to close the plot from the terminal
def plot():
	plt.ion()
	plt.show()
	warnings.filterwarnings("ignore",".*GUI is implemented.*")
	try:
		while (True):
			plt.pause(.02)
			if not plt.fignum_exists(1):
				return
	except KeyboardInterrupt:
		return

	# input('\n\tPress ENTER to exit\n')
	plt.close('all')

# Formats axis limits and visuals of the plot, and draws everything
def format_plot(path, smooth_path, turns):
	fig = plt.figure(1)
	ax = plt.axes()
	draw_path_arrows(path)
	draw_turns(path, turns)

	plt.axis('equal')
	min_x = np.min(path[:,0])
	max_x = np.max(path[:,0])
	min_y = np.min(path[:,1])
	max_y = np.max(path[:,1])

	plt.axis([min_x-2, max_x+2, min_y-2, max_y+2])
	plt.axis('off')
	fig.subplots_adjust(top = 1, bottom = 0, left = 0, right = 1)
	plt.plot(smooth_path[:,0], smooth_path[:,1], color = (0,0,1,.3))
	plot()

def format_piechart(times):
	times = np.array(times)
	times_percent = times/np.sum(times)*100
	labels = ['Generation', 'Interpolation', 'Smoothing', 'Turn calculation', 
		'Turn filtering']
	max_index = np.argmax(times)
	explode = tuple([0.05*int(i == max_index) for i, t in enumerate(times)])
	fig = plt.figure(2)
	ax = plt.axes()
	ax.pie(times_percent, 
		explode = explode, 
		labels = labels, 
		autopct = '%1.1f%%', 
		startangle = 90,
		counterclock = False)
	plt.axis('equal')

# Manages the algorithm and 
def setup(detail):
	print('\n\tGenerating path...')
	t0 = time.clock()
	# path = generate_easy_path(20)
	path = generate_brownian_path(200)
	t1 = time.clock()
	generate_time = t1-t0
	print('\tPath generated in %.1fms!' % (1e3*generate_time,))
	print('\tInterpolating points...')
	t2 = time.clock()
	interpolated_path = interpolate_path(path, detail)
	t3 = time.clock()
	interpolation_time = t3-t2
	print('\tPoints interpolated in %.1fms!' % (1e3*interpolation_time,))
	print('\tSmoothing path...')
	t4 = time.clock()
	smooth_path = filter_path(interpolated_path, detail)
	t5 = time.clock()
	smoothing_time = t5-t4
	print('\tPath smoothed in %.1fms!' % (1e3*smoothing_time,))
	print('\tCalculating turns...')
	t6 = time.clock()
	all_turns = calculate_turns(smooth_path, detail)
	# all_turns = calculate_turns_old(smooth_path, detail)
	t7 = time.clock()
	turn_calculation_time = t7-t6
	print('\tTurns calculated in %.1fms!' % (1e3*turn_calculation_time,))
	print('\tFiltering turns...')
	t8 = time.clock()
	turns = filter_turns(all_turns, detail)
	t9 = time.clock()
	turn_filtering_time = t9-t8
	print('\tTurns filtered in %.1fms!' % (1e3*turn_filtering_time,))
	total_time = interpolation_time + smoothing_time
	total_time += turn_calculation_time + turn_filtering_time
	print('\tTotal time: %.1fms' % (1e3*total_time,))
	print('\tPlotting path...')
	# plt.plot(smooth_path[:,0], smooth_path[:,1])
	format_piechart([generate_time, interpolation_time, smoothing_time, 
		turn_calculation_time, turn_filtering_time])
	format_plot(path, smooth_path, turns)
	print('\tAll done! Goodbye!\n')

if __name__ == '__main__':
	setup(int(sys.argv[1]) if len(sys.argv) == 2 else 10)