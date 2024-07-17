'''
LC 46, permuation， 
'''


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(path):
            if len(path) == len(nums):
                permutations.append(path[:])
                return

            for num in nums:
                if num not in path:
                    path.append(num)
                    backtrack(path)
                    path.pop()

        permutations = []
        backtrack([])
        return permutations

# follow up， follow-up 每个数可以使用多次，但至少一次，给定组合的长度（大于可选数的数量，比如给4个数然后排列长度为5），问所有的排列)

class Solution:
    def permute(self, nums: List[int],k) -> List[List[int]]:
        def backtrack(path, counts):
            if (len(path) == k) and (set(path)==set(nums)):
                permutations.append(path[:])
                return
            
            for num in nums:
                # 一个nums 最多用 k-n+1 次
                if counts[num] <=(k-n+1):
                    path.append(num)
                    counts[num] += 1
                    backtrack(path, counts)
                    path.pop()
                    counts[num] -= 1

        permutations = []
        # Initialize counts dictionary to track occurrences of each number in the path
        counts = {num: 0 for num in nums}
        backtrack([], counts)
        return permutations
