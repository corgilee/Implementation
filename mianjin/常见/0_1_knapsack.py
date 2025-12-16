

#https://www.youtube.com/watch?v=k8p0-JPbPJs

'''
dp is a 2D array where dp[i][w] represents the maximum value that can be obtained with the first i items 
and a total weight limit of w.

dp[i][j] means 在[0,i] 的物品里任取， 放进容量为 j 的背包里
'''

def knapsack(values, weights, capacity):
    n = len(values)
    # 先创建矩阵, 矩阵的 行是物品的数量，列是容量加1
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n)]

    '''
    取以下两者的最大值
    不取物品 i， dp[i-1][j]
    取 物品 i，dp[i-1][j-weights[i]]+values[i]
    '''
    # 初始化, 第一列都是0, 第一行，要根据第一个物品的重量来定
    for j in range(capacity+1):
        if j>=weights[0]:
            dp[0][j]=values[0]

    #print(dp)
    for i in range(n):
        for j in range(1,capacity+1):
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-weights[i]]+values[i])

    return dp[n-1][capacity]

# Example usage
values = [60, 100, 120]  # The values of the items
weights = [10, 20, 30]   # The weights of the items
capacity = 50            # Maximum capacity of the knapsack
print(knapsack(values, weights, capacity))
