# Example usage:
grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]


def numIslands(grid):
    # calculate and store the area of each island first
    #meta data
    m=len(grid)
    n=len(grid[0])
    dirs=((-1,0),(1,0),(0,1),(0,-1))
    areas=[]


    def dfs(i,j):
        if i<0 or i>=m or j<0 or j>=n or grid[i][j]!="1":
            return 0
        grid[i][j]="X" #先把i,j mark掉
        size=1

        size+=dfs(i+1,j)
        size+=dfs(i-1,j)
        size+=dfs(i,j+1)
        size+=dfs(i,j-1)

        return size

    for i in range(m):
        for j in range(n):
            if grid[i][j]=="1":
                areas.append(dfs(i,j))
    
    #areas=[4,3,1,2]
    def quickselect(l,r,nums,k):
        pivot=l
        base=nums[r]
        for i in range(l,r):
            if nums[i]<=base:
                nums[i],nums[pivot]=nums[pivot],nums[i]
                pivot+=1
        nums[pivot],nums[r]=nums[r],nums[pivot]
        if pivot==k:
            return nums[pivot]
        elif pivot>k:
            return quickselect(l,pivot-1, nums,k)
        else:
            return quickselect(pivot+1, r, nums,k)

    n=len(areas)
    if n%2==1:
        return quickselect(0,n-1,areas,n//2)
    else:
        return (quickselect(0,n-1,areas,n//2)+quickselect(0,n-1,areas,n//2+1))*0.5


print(numIslands(grid))  # Output should be the median size of the islands