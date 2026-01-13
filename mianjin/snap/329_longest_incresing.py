class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        
        #wenjie solution
        
        '''
        dfs+memo {}, memo {} 记得是每一个点能继续走的path
        T/S: O(m*n)
        '''

        if not matrix: return 0
        m, n = len(matrix), len(matrix[0])
        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        memo = {}

        def dfs(i, j):
            # 如果算过了，直接返回
            if (i, j) in memo:
                return memo[(i, j)]
            
            # 初始化：当前格子本身长度为 1
            max_len = 1
            
            for di, dj in dirs:
                ii, jj = i + di, j + dj
                # 严格递增条件：matrix[ii][jj] > matrix[i][j]
                if 0 <= ii < m and 0 <= jj < n and matrix[ii][jj] > matrix[i][j]:
                    max_len = max(max_len, 1 + dfs(ii, jj))
            
            # 存入记忆化字典
            memo[(i, j)] = max_len
            return max_len

        # 对每一个点调用 DFS，取最大值
        ans=0
        for i in range(m):
            for j in range(n):
                ans=max(ans,dfs(i, j))

        return ans


    ''''    
    # follow_up
       1） 如果不是严格increasing, 可以equal，怎么改； 
       2）要求打出path 两个followup都要求在原代码基础上改写，能跑
    '''

class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix: return 0
        m, n = len(matrix), len(matrix[0])
        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        memo = {}
        visited = set() # 用于处理非递减路径时的死循环

        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            # 基础路径就是自己
            best_path = [matrix[i][j]]
            visited.add((i, j)) # 标记当前路径已访问
            
            for di, dj in dirs:
                ii, jj = i + di, j + dj
                # 修改点 1：使用 >= 允许相等
                # 修改点 2：增加 visited 判断防止死循环
                if 0 <= ii < m and 0 <= jj < n and (ii, jj) not in visited and matrix[ii][jj] >= matrix[i][j]:
                    sub_path = dfs(ii, jj)
                    # 如果后续路径 + 当前点 长度更长，则更新
                    if len(sub_path) + 1 > len(best_path):
                        best_path = [matrix[i][j]] + sub_path
            
            visited.remove((i, j)) # 回溯，清理标记
            memo[(i, j)] = best_path
            return best_path

        # 寻找全局最长路径列表
        result_path = []
        for r in range(m):
            for c in range(n):
                path = dfs(r, c)
                if len(path) > len(result_path):
                    result_path = path
        
        # 打印完整路径
        print(f"Follow-up Path: {result_path}")
        return len(result_path)