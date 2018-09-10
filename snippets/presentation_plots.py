from calculate_turns import *
import matplotlib
# Remove this if your system doesn't have support for Qt5
matplotlib.use("Qt5agg")
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm
import time
import random
from scipy.ndimage.filters import gaussian_filter
import sys
import warnings

path = generate_easy_path(20)
detail = 3
inter = interpolate_path(path, detail)
smooth = filter_path(inter, detail)
turns = calculate_turns(smooth, detail)
plt.plot(smooth[:,0], smooth[:,1], color=(0,.5,0,.5))
plt.plot(inter[:,0], inter[:,1], 'b.') 
plt.plot(path[:,0], path[:,1], 'r.', ms=10)
# fig = plt.figure(1)
# plt.axis('off')
# fig.subplots_adjust(top = 1, bottom = 0, left = 0, right = 1)
# plt.shsow()
format_plot(path, smooth, turns)