'''
https://www.cnblogs.com/labuladong/p/13937987.html

'''

# option 1: dp, dp[k][n] means the necessary steps with current k eggs and n floors
class Solution:
    # time complexity: O(k*n^2)
    def superEggDrop(self, k: int, n: int) -> int:
        memo = dict()
        def dp(K, N) -> int:
            # base case
            if K == 1: return N 
            if N == 0: return 0
            # 避免重复计算
            if (K, N) in memo:
                return memo[(K, N)]

            res = float('INF')
            # 穷举所有可能的选择
            for i in range(1, N + 1):
                res = min(res, max(dp(K, N - i), dp(K - 1, i - 1)) + 1 )
            # 记入备忘录
            memo[(K, N)] = res
            return res
    
        return dp(k, n)

# option2: dp:  #dp[i][j] represents the maximum number of floors that can be checked with i eggs and j trials.
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:

        #dp[i][j] represents the maximum number of floors that can be checked with i eggs and j trials.
        dp = [[0]*(n+1) for _ in range(k+1)] # i is #egg, j is trials,j的最大值也就是n了
        
        for j in range(1, n+1):
            for i in range(1, k+1):
                '''
                1、无论你在哪层楼扔鸡蛋，鸡蛋只可能摔碎或者没摔碎，碎了的话就测楼下，没碎的话就测楼上。
                2、无论你上楼还是下楼，总的楼层数 = 楼上的楼层数 + 楼下的楼层数 + 1（当前这层楼）
                '''
                dp[i][j] = dp[i][j-1] + dp[i-1][j-1] + 1

                '''
                dp[k][m - 1] 就是楼上的楼层数，因为鸡蛋个数 k 不变，也就是鸡蛋没碎，扔鸡蛋次数 m 减一；
                dp[k - 1][m - 1] 就是楼下的楼层数，因为鸡蛋个数 k 减一，也就是鸡蛋碎了，同时扔鸡蛋次数 m 减一。
                如果dp[k-1][m-1]=6, dp[k][m-1]=2,那么就是 6+1+2
                dp[k][m-1]=2 不代表只能从0-2，代表的是任意base，能测[base,base+2]
                '''
                
                # If the minimum number of moves exceeds n, return the current floor.
                if dp[i][j] >= n:
                    return j
        return -1