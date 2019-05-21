#!/usr/bin/env python3
import pickle
import requests
import time

a = {'hello': 'world'}

with open('filename.pickle', 'wb') as handle:
	pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
	b = pickle.load(handle)

def get_subreddit_posts(**kwargs):
	r = requests.get("https://api.pushshift.io/reddit/search/submission/",params=kwargs)
	data = r.json()
	return data['data']

try:
	with open('reddit_scraper.pickle', 'rb') as handle:
		b = pickle.load(handle)
		before = b["before"]
		count = b["count"]
except:
	print("Starting from scratch")
	before = None
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
		pickle.dump({"before": before, "count": count}, 
			handle, protocol=pickle.HIGHEST_PROTOCOL)
	post = posts[-1]
	print(
		post['id'], 
		count, 
		"%.1f days scraped" % ((time.time() - post['created_utc']) / 86400,), 
		end = "\t", flush = True
	)

	with open('reddit_post_urls.txt', 'a') as f:
		for post in posts:
			# before = post['created_utc'] # This will keep track of your position for the next call in the while loop
			# Do stuff with each comment object
			# Example (print comment id, epoch time of comment and subreddit and score)
			# print(post)
			f.write("%s\n" % (post["url"],))
			# print(post['id'],post['created_utc'],post['subreddit'],post['score'])

	print("Safe")
	time.sleep(.1)