from random import random
import math

s = 9.9999
n = 10
nums = [-math.log(1-random()) for i in range(n)]
sums = sum(nums)
nums = [i*s/sums for i in nums]
print(nums)
