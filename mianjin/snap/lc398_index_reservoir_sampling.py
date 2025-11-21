'''

coding 挂，印度小哥哥问了一道random index 的题，没做出来，reservoir sampling 那种 (类似 lc 398)

没有任何coding技巧。原理就是reservoir sampling，重点在于要求现场证明一下为什么你的办法可行

'''

import random

class Solution:
    def __init__(self, nums):
        self.nums = nums

    def pick(self, target):
        count = 0   # count how many target values seen
        ans = -1    # current chosen index
        
        for i, num in enumerate(self.nums):
            if num == target:
                count += 1
                # Reservoir sampling: pick with probability 1/count
                if random.randint(1, count) == 1:
                    ans = i
        return ans


nums = [1, 2, 3, 3, 3]
solution = Solution(nums)

result_count = {2: 0, 3: 0, 4: 0}

# Run pick() many times to check distribution
for _ in range(10000):
    idx = solution.pick(3)
    result_count[idx] += 1

print(result_count)

'''
Explanation step-by-step:

Scan the array one element at a time.

Every time you see a target, increase count:
count = how many targets we have seen so far

With probability 1/count, replace the previously chosen answer with the new index.

This ensures:

the 1st occurrence is chosen with 100% at the moment
the 2nd occurrence replaces the 1st with prob 1/2
the 3rd replaces the previous with prob 1/3

the k-th replaces previous with prob 1/k
After finishing the scan, ans is a uniformly random index from all valid indices.

'''

'''
Proof Intuition:
For the j-th occurrence to become the final answer, two things must happen:

1. It must be selected at the moment it appears.

When we encounter the j-th target, it is chosen with probability:
1 / j

2. It must NOT be replaced by any later occurrence.

When processing the (j+1)-th occurrence:
chance of replacing the current selection = 1/(j+1)
chance of not replacing = j/(j+1)

Similarly:
(j+2)-th occurrence: probability of not replacing = (j+1)/(j+2)
…
k-th occurrence: probability of not replacing = (k-1)/k

Multiply all these probabilities:

P(the j-th occurrence is the final selected index)
= (1/j)
  × (j/(j+1))
  × ((j+1)/(j+2))
  × …
  × ((k-1)/k)
  =1/k

'''