class Solution:
    def trap(self, height: List[int]) -> int:
        '''
        the water in each slot depends on the hight difference between the current bar
        and the min(highest bar on the left side, highest bar on the right side)
        '''
        res=0
        n=len(height)
        left=[0]*n
        right=[0]*n
        current=[0]*n
        for i in range(1,n):
            left[i]=max(left[i-1],height[i-1])
        for i in range(n-2,-1,-1):
            right[i]=max(right[i+1],height[i+1])

        for i in range(n):
            current[i]=min(left[i],right[i])
            res+=max(current[i]-height[i],0)

        return res