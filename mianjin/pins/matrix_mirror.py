'''

'''


def can_reach_end(matrix, start, end):
    m, n = len(matrix), len(matrix[0])
    visited = [[False] * n for _ in range(m)]
    
    def dfs(i, j, direction):
        if i < 0 or i >= m or j < 0 or j >= n or visited[i][j]:
            return False
        
        visited[i][j] = True
        
        if (i, j) == end:
            return True
        
        if matrix[i][j] == 45:
            if direction == 'up':
                return dfs(i - 1, j, 'left')
            elif direction == 'down':
                return dfs(i + 1, j, 'right')
            elif direction == 'left':
                return dfs(i, j - 1, 'up')
            elif direction == 'right':
                return dfs(i, j + 1, 'down')
        
        if matrix[i][j] == 135:
            if direction == 'up':
                return dfs(i - 1, j, 'right')
            elif direction == 'down':
                return dfs(i + 1, j, 'left')
            elif direction == 'left':
                return dfs(i, j - 1, 'down')
            elif direction == 'right':
                return dfs(i, j + 1, 'up')
        
        return dfs(i - 1, j, 'up') or dfs(i + 1, j, 'down') or dfs(i, j - 1, 'left') or dfs(i, j + 1, 'right')
    
    return dfs(start[0], start[1], 'up')


    
matrix = [
  [0, 0, 0, 0],
  [0, 45, 0, 0],
  [0, 0, 135, 0],
  [0, 0, 0, 0]
]
start = [1, 1]
end = [2, 2]

print(can_reach_end(matrix, start, end))