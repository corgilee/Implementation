class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        sorted_tasks=[]
        for index, val in enumerate(tasks):
            sorted_tasks.append([val[0],val[1],index])
        sorted_tasks.sort() #nlog(n)

        res=[]
        task_index=0 #use this index to point the task needs to added to the waitlist
        c_time=0
        waitlist=[]
        while len(res)<len(tasks):
            # 如果当前waitlist没有task，把当前时间快进到下一个项目的开始时间
            if len(waitlist)==0 and c_time<sorted_tasks[task_index][0]:
                c_time=sorted_tasks[task_index][0]

            while task_index<len(tasks) and sorted_tasks[task_index][0]<=c_time:
                _, process_time, index=sorted_tasks[task_index]
                heapq.heappush(waitlist,(process_time,index))
                task_index+=1
            
            #这个时候waitlist肯定是有东西的，那么pop 出来放到res里面
            process_time, index=heapq.heappop(waitlist)
            
            res.append(index)
            # time 直接快进到当前任务结束
            c_time+=process_time 

        return res