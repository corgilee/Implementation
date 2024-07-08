class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        '''
        要用一个visited 储存(i,j,k),储存k很重要，这样保证里i，j可以用不同的strategy，
        最短路径用bfs
        起点(0,0),终点(m-1,n-1), output 是step
        #T: O(m*n*k)
        '''
        m, n = len(grid), len(grid[0])
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        target = (m-1, n-1)

        # edge case: (m+n-2)
        # 如果可以清理足够多的障碍物，那么最短距离就是m+n-2
        if k>=m+n-2:
            return m+n-2

        state=(0,0,k) #i,j,k
        visited=set()
        visited.add(state)

        q=collections.deque()
        q.append([0,state]) #step,i,j,k

        while q:
            step,(i,j,k)=q.popleft()
            if (i,j)==target:
                return step
            
            for di,dj in dirs:
                ii=i+di
                jj=j+dj
                if ii>=0 and ii<m and jj>=0 and jj<n:
                    new_k=k-grid[ii][jj]
                    new_state=(ii,jj,new_k)
                    if new_k>=0 and new_state not in visited:
                        visited.add(new_state)
                        q.append([step+1,new_state])

        return -1

        

        
 