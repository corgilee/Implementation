class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        '''
        要用一个visited 储存(i,j,k),储存k很重要，这样保证里i，j可以用不同的strategy，最短路径用bfs
        起点(0,0),终点(m-1,n-1), output 是step
        #T: O(m*n*k)
        '''
        rows, cols = len(grid), len(grid[0])
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        target = (rows-1, cols-1)
        
        if k>=rows+cols-2:
            return rows+cols-2

        state = (0, 0, k)
        seen = set()
        seen.add(state)
        queue = deque()
        queue.append([0, state])

        while queue:
            steps, (row, col, k) = queue.popleft()
            if (row, col) == target:
                return steps

            for dr, dc in directions:
                new_row, new_col = row+dr, col+dc
                if 0<=new_row<rows and 0<=new_col<cols:
                    new_eliminate = k-grid[new_row][new_col] 
                    #上面这样写的话就可以少去了很多if
                    new_state = (new_row, new_col, new_eliminate)
                    if new_eliminate>=0 and new_state not in seen:
                        seen.add(new_state)
                        queue.append([steps+1, new_state])

        return -1
