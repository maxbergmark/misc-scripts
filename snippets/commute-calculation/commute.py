#!/usr/bin/env python3
import pickle
import requests
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import math
import random
import colorsys

CBK = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472]
CEK = [0.7111111111111111, 1.4222222222222223, 2.8444444444444446, 5.688888888888889, 11.377777777777778, 22.755555555555556, 45.51111111111111, 91.02222222222223, 182.04444444444445, 364.0888888888889, 728.1777777777778, 1456.3555555555556, 2912.711111111111, 5825.422222222222, 11650.844444444445, 23301.68888888889, 46603.37777777778, 93206.75555555556, 186413.51111111112, 372827.02222222224, 745654.0444444445, 1491308.088888889, 2982616.177777778, 5965232.355555556, 11930464.711111112, 23860929.422222223, 47721858.844444446, 95443717.68888889, 190887435.37777779, 381774870.75555557, 763549741.5111111]
CFK = [40.74366543152521, 81.48733086305042, 162.97466172610083, 325.94932345220167, 651.8986469044033, 1303.7972938088067, 2607.5945876176133, 5215.189175235227, 10430.378350470453, 20860.756700940907, 41721.51340188181, 83443.02680376363, 166886.05360752725, 333772.1072150545, 667544.214430109, 1335088.428860218, 2670176.857720436, 5340353.715440872, 10680707.430881744, 21361414.86176349, 42722829.72352698, 85445659.44705395, 170891318.8941079, 341782637.7882158, 683565275.5764316, 1367130551.1528633, 2734261102.3057265, 5468522204.611453, 10937044409.222906, 21874088818.445812, 43748177636.891624]


def ll2px(lat, lng, zoom):
	"""Given two floats and an int, return a 2-tuple of ints.
	Note that the pixel coordinates are tied to the entire map, not to the map
	section currently in view.
	"""
	assert isinstance(lat, (float, int)), \
		ValueError("lat must be a float")
	lat = float(lat)
	assert isinstance(lng, (float, int)), \
		ValueError("lng must be a float")
	lng = float(lng)
	assert isinstance(zoom, int), TypeError("zoom must be an int from 0 to 30")
	assert 0 <= zoom <= 30, ValueError("zoom must be an int from 0 to 30")

	cbk = CBK[zoom]

	x = int(round(cbk + (lng * CEK[zoom])))

	foo = math.sin(lat * math.pi / 180)
	if foo < -0.9999:
		foo = -0.9999
	elif foo > 0.9999:
		foo = 0.9999

	y = int(round(cbk + (0.5 * math.log((1+foo)/(1-foo)) * (-CFK[zoom]))))

	return (x, y)

api_key = "AIzaSyCphPai8-x6SVgCWXyJbo9NHnugdcGg8FE"
directions_base_url = "https://maps.googleapis.com/maps/api/directions/json?"
map_base_url = "http://maps.googleapis.com/maps/api/staticmap?"

def get_direction(origin, destination):
	options = {
		"origin": "%f,%f" % origin,
		"destination": "%f,%f" % destination,
		"key": api_key,
		"mode": "transit"
	}
	url = directions_base_url + "&".join([k+"="+str(v) for k, v in options.items()])
	response = requests.get(url)
	response_json = response.json()
	# print(response_json["routes"][0])
	if len(response_json["routes"]) > 0:
		route = response_json["routes"][0]
		if len(route["legs"]) > 0:
			duration = route["legs"][0]["duration"]["value"]
			return duration / 60
	return 0

def check_if_water(position):
	map_options = {
		"center": "%f,%f" % position,
		"zoom": 20, 
		"size": "1x1",
		"maptype": "roadmap",
		"sensor": "false",
		"key": api_key
	}

	url = map_base_url + "&".join([k+"="+str(v) for k, v in map_options.items()])
	try:
		resp = requests.get(url, timeout = 10)
		image = np.asarray(bytearray(resp.content), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		return np.array_equal(image[0][0], [255, 219, 171])
	except:
		print("Error")

def show_entire_map(coords):
	# position = "59.320614,18.035773"
	# position = "59.336688,17.913016"
	# position = "59.384314,18.166843"
	# position = "59.250048375,17.946944375"
	# position = "59.425829625,17.946944375"
	center = (59.337939, 18.034835)
	zoom = 11
	dim = 640
	map_options = {
		"center": "%f,%f" % center,
		"zoom": zoom, 
		"size": "%dx%d" % (dim, dim),
		# "maptype": "satellite",
		"sensor": "false",
		"key": api_key
	}


	# pos_x, pos_y = (float(i) for i in position.split(","))
	# center_x, center_y = (float(i) for i in center.split(","))
	center_pixel = ll2px(center[0], center[1], zoom)
	url = map_base_url + "&".join([k+"="+str(v) for k, v in map_options.items()])
	try:
		resp = requests.get(url, timeout = 10)
		image = np.asarray(bytearray(resp.content), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	except Exception as e:
		print("Error:", e)
	patches = []
	for pos in coords:
		pixel = ll2px(pos["coord"][0], pos["coord"][1], zoom)
		diff_x, diff_y = pixel[0]-center_pixel[0], pixel[1]-center_pixel[1]
		col = colorsys.hsv_to_rgb((pos["travel_time"]//10)/6, 1, 1)
		color = (col[0], col[1], col[2], .3)
		patches.append(Circle((dim/2 + diff_x, dim/2 + diff_y), radius=5, color=color))
	fig, ax = plt.subplots(1)
	ax.imshow(image)
	for p in patches:
		ax.add_patch(p)
	plt.show(fig)

def get_random_coordinates(n):
	# min_sample = np.array([59.301719, 17.987629])
	min_sample = np.array([59.253941, 17.857175])
	# max_sample = np.array([59.358786, 18.141519])
	max_sample = np.array([59.440819, 18.257332])
	points = np.random.random((n, 2))
	points *= max_sample - min_sample
	points += min_sample
	return [tuple(p) for p in points.tolist()]



coords = get_random_coordinates(500)
filtered_coords = [{"coord": c} for c in coords if not check_if_water(c)]
destination = (59.332483, 18.062642)

for c in filtered_coords:
	c["travel_time"] = get_direction(c["coord"], destination)

filtered_times = [c for c in filtered_coords if c["travel_time"] > 0]

show_entire_map(filtered_times)
