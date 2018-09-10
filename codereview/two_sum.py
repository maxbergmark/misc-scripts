import time

class Solution:
    def twoSum(self, nums, target):
        number_bonds = {}
        for index, value in enumerate(nums):
            if value in number_bonds:
                return [number_bonds[value], index]
            number_bonds[target - value] = index
        return None
    def twoSum3(self, nums, target):
        number_bonds = set()
        for index, value in enumerate(nums):
            if value in number_bonds:
                return True
            number_bonds.add(target - value)
        return None
    def twoSum2(self, numbers, target):
        bounds = {target - value: index for index, value in enumerate(numbers)}
        return next(((index, bounds[value]) for index, value in enumerate(numbers) if value in bounds), None)
nums = [i**2 for i in range(101)]
target = 19801
n = 10000
t0 = time.clock()
for i in range(n):
	s = Solution().twoSum(nums, target)
t1 = time.clock()
for i in range(n):
	s2 = Solution().twoSum2(nums, target)
t2 = time.clock()
print(s, s2, t1-t0, t2-t1, (t1-t0)/(t2-t1))