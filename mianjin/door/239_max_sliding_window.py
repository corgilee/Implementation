class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        '''
        因为对顺序有讲究，所以用 queue 来保存index
        queue 
        1. 先进先出，不可能后加入的在新加入的前面
        2. 通过while 循环，当前情况下最大值的肯定在首位
            q.popleft(),q.pop() #可以pop左边，也可以pop右边
        
        T: O(k+n-k)=O(n), q的popleft,pop 复杂度都是O(1)
        '''
        res=[]
        q=collections.deque()
        for i in range(k):
            while q and nums[i]>nums[q[-1]]:
                q.pop()
            #每次都要append
            q.append(i) #存入的是index，不是value
        
        res.append(nums[q[0]])

        for i in range(k,len(nums)):
            #每次新加入一个new value,要看一下队列第一个需不需要弹出
            if q and q[0]==i-k:
                q.popleft()

            while q and nums[i]>=nums[q[-1]]:
                q.pop()
            
            # 最后一定要加入
            q.append(i)
            #把队列首位放到res里面
            res.append(nums[q[0]])

        return res



