'''
给一些ints比如【1，3，7】，和一个target，问至少从ints里面取多少个能加起来等于target，可以重复取，如果不能组成返回特殊值, could you implement this in python？
'''

def min_elements_to_sum(target, nums):
    # Initialize the dp array with infinity (a large number)
    dp = [float('inf')] * (target + 1)
    # dp[i] 代表的是 当total_sum 为i 时 的可能组合数
    dp[0] = 0  # Base case: it takes 0 elements to sum up to 0

    for i in range(1, target + 1):
        # O(target)
        for num in nums:
            # O(n)
            if i >= num:
                dp[i] = min(dp[i], dp[i - num] + 1)

    return dp[target] if dp[target] != float('inf') else -1