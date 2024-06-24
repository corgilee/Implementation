    def numIslands(self, grid: List[List[str]]) -> int:
        '''
        两种解法 
        bfs:q=collections.deque(), + visited=set()
        dfs 每到一个点就把他搞成”X“, 
        O(m*n)
        '''
        #meta
        m=len(grid)
        n=len(grid[0])
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        res=0

        def dfs(i,j):
            grid[i][j]="x"
            for di,dj in dirs:
                ii=di+i
                jj=dj+j
                if ii>=0 and ii<m and jj>=0 and jj<n and grid[ii][jj]=="1":
                    dfs(ii,jj)

        # main function
        for i in range(m):
            for j in range(n):
                if grid[i][j]=="1":
                    dfs(i,j)
                    res+=1
        return res
