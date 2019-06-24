import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

n = 366
data = []
cols = [[] for i in range(n)]
avgs = np.zeros((n, 3))
limit = 1000000
i = 0
with open("image_parser_results.txt", "r") as f:
	for line in f:
		if (line.count("22.33589744") == 3):
			continue
		d = ast.literal_eval(line.replace("array(", "").replace("))", ")"))
		if (1.5*d[2][1] > d[2][0]+d[2][2]) or (1.5*d[2][0] > d[2][1]+d[2][2]) or (1.5*d[2][2] > d[2][0]+d[2][1]):
			data.append(d)
		if i == limit:
			break
		i += 1
		# print(data[-1])

for timestamp, url, col in data:
	date = datetime.fromtimestamp(timestamp)
	day_of_year = date.timetuple().tm_yday
	cols[day_of_year-1].append(col)
	# cols[(timestamp % 86400)// 60].append(col)

for day in range(len(cols)):
	cols[day] = np.array(cols[day])
	# print(cols[day])
	avgs[day] = np.mean(cols[day], axis = 0)

avgs /= 255
# print(avgs)

fig = plt.figure()

display_axes = fig.add_axes([0.1,0.1,0.8,0.8], projection='polar')
display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
								  ## multiply the values such that 1 become 2*pi
								  ## this field is supposed to take values 1 or -1 only!!

norm = mpl.colors.Normalize(0.0, 2*np.pi)

# Plot the colorbar onto the polar axis
# note - use orientation horizontal so that the gradient goes around
# the wheel rather than centre out

# colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
# n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(
		cmap_name, avgs, N=n)

cb = mpl.colorbar.ColorbarBase(display_axes, cmap = cm,
								   norm=norm,
								   orientation='horizontal')

# aesthetics - get rid of border and axis labels								   
cb.outline.set_visible(False)								 
display_axes.set_axis_off()
plt.show() # Replace with plt.savefig if you want to save a file