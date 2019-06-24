#!/usr/bin/env python3
import pickle
import requests
import time
import json

def get_subreddit_posts(**kwargs):
	r = requests.get("https://api.pushshift.io/reddit/search/submission/",params=kwargs)
	data = r.json()
	return data['data']

try:
	with open('reddit_scraper.pickle', 'rb') as handle:
		b = pickle.load(handle)
		start_time = b["start_time"]
		before = b["before"]
		count = b["count"]
except:
	print("Starting from scratch")
	before = None
	start_time = time.time()
	count = 0

while True:
	print("Fetching", end = "\t", flush = True)
	posts = get_subreddit_posts(
		subreddit = "pics", 
		size = 500, 
		before = before, 
		sort = 'desc', 
		sort_type = 'created_utc'
	)
	if not posts:
		break
	print("Saving", end = "\t", flush = True)

	before = posts[-1]['created_utc']
	count += 1
	with open('reddit_scraper.pickle', 'wb') as handle:
		pickle.dump({"before": before, "count": count, "start_time": start_time}, 
			handle, protocol=pickle.HIGHEST_PROTOCOL)
	post = posts[-1]
	print(
		"id: %s\tcount: %d\t%.1f / %.1f days scraped" % (
			post['id'], 
		 	count, 
			(start_time - post['created_utc']) / 86400, 
			(start_time - 1201223694) / 86400
		), 
		end = "\t", flush = True
	)

	with open('reddit_post_jsons.txt', 'a') as f:
		for post in posts:
			# before = post['created_utc'] # This will keep track of your position for the next call in the while loop
			# Do stuff with each comment object
			# Example (print comment id, epoch time of comment and subreddit and score)
			# print(post)
			f.write("%s\n" % (json.dumps(post),))
			# print(post['id'],post['created_utc'],post['subreddit'],post['score'])

	print("Safe")
	time.sleep(.1)