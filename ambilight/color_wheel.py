# from urllib import request, parse
import requests
import json
import time
import random
import colorsys

def generate_color_json(colors, c, n, res):
	left = {}
	right = {}
	for i in range(n//2):
		left[str(i)] = {"r":colors[(c-res*i)%(n*res)][0],
						"g":colors[(c-res*i)%(n*res)][1],
						"b":colors[(c-res*i)%(n*res)][2]}
		right[str(i)] = {"r":colors[(c-res*(i+n//2))%(n*res)][0],
						 "g":colors[(c-res*(i+n//2))%(n*res)][1],
						 "b":colors[(c-res*(i+n//2))%(n*res)][2]}
	return {"layer1":{"left":left, "right": right}}

n = 10
res = 5
colors = [colorsys.hsv_to_rgb(i/(n*res), 1, 1) for i in range(n*res)]
colors = [[int(255*i) for i in c] for c in colors]

apikey = 'fFFffSTuu1CTj8WBucsNL8FzD8Rv0Lhy'
endpoint = '/tv/1/ambilight/cached'
url = 'https://maxbergmark.duckdns.org:4433%s?apikey=%s' % (endpoint, apikey)
c = 0
t0 = time.time()

while True:
	c += 1
	data = json.dumps(generate_color_json(colors, c, n, res))
	req = requests.post(url, data=data)
	elapsed = time.time()-t0
	time.sleep(max(0, 0.3-elapsed))
	t0 = time.time()