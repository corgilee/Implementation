import random
class Solution:

    def __init__(self, w: List[int]):
        '''
        1. presum
        2. random.choice range(1,presum+1)
        3. 用类似binary search 可以log(n), iterate 的话是 O(n)
        '''
        self.n=len(w)
        self.presum=[0]*self.n
        self.presum[0]=w[0]
        for i in range(1,self.n):
            self.presum[i]=self.presum[i-1]+w[i]

    def pickIndex(self) -> int:
        random_num=random.choice(range(1,self.presum[-1]+1))
        l,r=0,len(self.presum)
        # find the smallest index where presum[mid]>random_num
        while l<=r:
            mid=(l+r)//2
            if self.presum[mid]<random_num:
                l=mid+1
            else:
                r=mid-1
        # 返回的是l，因为l能代表最小的符合条件的index
        # 不是返回mid (只有在while中找target 才返回mid)
        return l