from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import requests
import time
# import cv2
import pickle
from datetime import datetime
import json
import urllib.request

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	# resp = urllib.urlopen(url)
	try:
		filename = url.split("/")[-1]
		# print("saving %s to %s" % (url, filename))
		directory = "/home/max/storage/images/"
		# directory = "images/"
		urllib.request.urlretrieve(url, directory + filename)
		# resp = requests.get(url, timeout = 10)
		# image = np.asarray(bytearray(resp.content), dtype="uint8")
		# image = cv2.imdecode(image, cv2.IMREAD_COLOR)

		# return the image
		# if (image is not None and len(image.shape) == 3):
			# return resp
			# avgs = np.mean(image, axis = (0, 1))
			# return avgs
	except Exception as e:
		# print("exception:", e)
		pass
	# return None

def parse_images(files):
	res = []
	for url in files:
		# print(image_data)
		# if image_data.count("\t") != 1:
			# continue
		# timestamp, url = image_data.split("\t")
		if url.endswith(".jpg") or url.endswith(".png") or url.startswith("https://i.reddituploads.com"):
			url_to_image(url)
			# if data is not None:
				# res.append((get_stamp(image_data), url, data))
		if url.startswith("https://imgur.com") or url.startswith("http://imgur.com"):
			continue
		# print(url)
	return res

def get_stamp(image_data):
	if image_data.count("\t") != 1:
		return -1
	timestamp, url = image_data.split("\t")
	if timestamp == "":
		return -1
	return int(timestamp)

def get_image_format(files, processes):
	with Pool(processes) as pool:
		results = pool.starmap(parse_images, files)
	return results

def process_batch(files, batch, batch_size, processes, data):
	splits = [[] for _ in range(processes)]
	for i in range(batch_size):
		splits[i%processes].append(files[batch*batch_size + i])
	for i in range(len(splits)):
		splits[i] = (splits[i],)
	t0 = time.time()
	colors = get_image_format(splits, processes)
	flat_colors = [c for i in colors for c in i]
	t1 = time.time()
	for c in flat_colors:
		data[str(c[0]) + c[1]] = c
		# print(c)
	print("%4.1f images/second" % (batch_size / (t1-t0),), end = "\t", flush = True)


try:
	with open('image_parser.pickle', 'rb') as handle:
		b = pickle.load(handle)
		start_time = b["start_time"]
		batch = b["batch"]
		# data = b["data"]
except:
	print("Starting from scratch")
	# data = {}
	start_time = time.time()
	batch = 0

# print("%d images already parsed" % len(data))
print("Starting from batch %d" % batch)

processes = cpu_count()*2
batch_size = 100
data = {}

files = []
# limit = 100

with open("reddit_post_jsons.txt", "r") as f:
	count = 0
	for l in f:
		files.append(json.loads(l.strip())["url"])
		count += 1
		# if count == limit:
			# break
	# files = [(l,) for l in f.read().split("\n")]

nr_lines = len(files)
tot_batches = nr_lines // batch_size

for _ in range(batch, nr_lines // batch_size + 1):
	print("%s\t%5d/%5d" % (
		datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
		batch, 
		tot_batches
	), end = "\t", flush = True)
	process_batch(files, batch, batch_size, processes, data)
	batch += 1
	# with open('image_parser_results.txt', 'a') as f:
		# for key in sorted(data.keys()):
			# f.write(str(data[key]) + "\n")
		# f.flush()
	data = {}
	with open('image_parser.pickle', 'wb') as handle:
		pickle.dump({"batch": batch, "start_time": start_time}, 
			handle, protocol = pickle.HIGHEST_PROTOCOL)
	print("SAFE")

