class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        '''
        把每个独立的island都改成统一编号(-1,-2,...)，然后面积算出来，存在一个dict里面 d={-1:20,-2:30}
        遍历grid，如果有0的话，查看周围编号和面积 记录下来和最大比较
        '''
        #meta data
        dirs=[(-1,0),(1,0),(0,-1),(0,1)]
        m=len(grid)
        n=len(grid[0])
        island_index=-1
        d={} #island: area
        
        def dfs(i,j):
            # 返回值是area
            if i<0 or i>=m or j<0 or j>=n or grid[i][j]!=1:
                return 0
            else:
                grid[i][j]=island_index # 把能到达的都index成一个编号
                area=1

                for di,dj in dirs:
                    ii=i+di
                    jj=j+dj
                    area+=dfs(ii,jj)
                return area

        #iterate the grid to mark index each island
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    area=dfs(i,j)
                    d[island_index]=area
                    island_index-=1
        
        # 遍历grid，每遇到grid[i][j]==0，就统计一下周围岛屿的编号，计算一下面积
        
        max_area=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==0:
                    area=1
                    surround_index=set()
                    for di,dj in dirs:
                        ii=di+i
                        jj=dj+j
                        if ii>=0 and ii<m and jj>=0 and jj<n and grid[ii][jj]<0:
                            surround_index.add(grid[ii][jj])
                    for index in surround_index:
                        area+=d[index]
                max_area=max(area,max_area)

        #edge case, all 1 cross the grid
        if max_area==0:
            return m*n

        return max_area